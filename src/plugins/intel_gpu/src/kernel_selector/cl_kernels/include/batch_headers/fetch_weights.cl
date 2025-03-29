// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common.cl"

#define GET_FILTER_OS_IS_YX_ISV_OSV_INDEX(prefix, o, i, y, x, osv, isv) \
    get_os_is_zyx_isv_osv_index(                                  \
        o, i, 0, y, x,                                            \
        CAT(prefix, _SIZE_X),                                     \
        CAT(prefix, _SIZE_Y),                                     \
        1,                                                        \
        CAT(prefix, _IFM_NUM),                                    \
        CAT(prefix, _OFM_NUM),                                    \
        osv,                                                      \
        isv                                                       \
    )

#define GET_FILTER_IS_OS_YX_OSV_ISV_INDEX(prefix, o, i, y, x, osv, isv) \
    get_os_is_zyx_isv_osv_index(                                  \
        i, o, 0, y, x,                                            \
        CAT(prefix, _SIZE_X),                                     \
        CAT(prefix, _SIZE_Y),                                     \
        1,                                                        \
        CAT(prefix, _OFM_NUM),                                    \
        CAT(prefix, _IFM_NUM),                                    \
        isv,                                                      \
        osv                                                       \
    )

#define GET_FILTER_IS_OS_YX_ISV_OSV_INDEX(prefix, o, i, y, x, osv, isv) \
    get_is_os_zyx_isv_osv_index(                                  \
        o, i, 0, y, x,                                            \
        CAT(prefix, _SIZE_X),                                     \
        CAT(prefix, _SIZE_Y),                                     \
        1,                                                        \
        CAT(prefix, _IFM_NUM),                                    \
        CAT(prefix, _OFM_NUM),                                    \
        osv,                                                      \
        isv                                                       \
    )

#define GET_FILTER_OS_IS_ZYX_ISV_OSV_INDEX(prefix, o, i, z, y, x, osv, isv) \
    get_os_is_zyx_isv_osv_index(                                  \
        o, i, z, y, x,                                            \
        CAT(prefix, _SIZE_X),                                     \
        CAT(prefix, _SIZE_Y),                                     \
        CAT(prefix, _SIZE_Z),                                     \
        CAT(prefix, _IFM_NUM),                                    \
        CAT(prefix, _OFM_NUM),                                    \
        osv,                                                      \
        isv                                                       \
    )

#define GET_FILTER_IS_OS_ZYX_ISV16_OSV16_INDEX(prefix, o, i, z, y, x, sub_group_size) \
    CAT(prefix, _OFFSET) +                                                            \
    ((o) % (sub_group_size)) +                                                        \
    (sub_group_size)*(                                                                \
        (x)*(sub_group_size)*CAT(prefix, _X_PITCH) +                                  \
        (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) +                                  \
        (z)*(sub_group_size)*CAT(prefix, _Z_PITCH) +                                  \
        ((i) % (sub_group_size)) +                                                    \
        ((o) / (sub_group_size))*(sub_group_size)*CAT(prefix, _OFM_PITCH) +           \
        ((i) / (sub_group_size))*CAT(prefix, _IFM_PITCH)                              \
    )

#define GET_FILTER_IS_OS_YX_ISV16_OSV16_INDEX(prefix, o, i, y, x, sub_group_size) \
    CAT(prefix, _OFFSET) +                                                        \
    ((o) % (sub_group_size)) +                                                    \
    (sub_group_size)*(                                                            \
        (x)*(sub_group_size)*CAT(prefix, _X_PITCH) +                              \
        (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) +                              \
        ((i) % (sub_group_size)) +                                                \
        ((o) / (sub_group_size))*(sub_group_size)*CAT(prefix, _OFM_PITCH) +       \
        ((i) / (sub_group_size))*CAT(prefix, _IFM_PITCH)                          \
    )

#define GET_FILTER_OS_IS_YX_ISV8_OSV16_ISV2_INDEX(prefix, o, i, y, x, sub_group_size) \
    get_os_is_zyx_isv8_osv16_isv2_index(                                              \
        0, o, i, 0, y, x,                                                             \
        CAT(prefix, _SIZE_X),                                                         \
        CAT(prefix, _SIZE_Y),                                                         \
        CAT(prefix, _SIZE_Z),                                                         \
        CAT(prefix, _GROUPS_NUM),                                                     \
        CAT(prefix, _OFM_NUM),                                                        \
        CAT(prefix, _IFM_NUM),                                                        \
        CAT(prefix, _OFFSET)                                                          \
    )

#define GET_FILTER_OS_IS_ZYX_ISV8_OSV16_ISV2_INDEX(prefix, o, i, z, y, x, sub_group_size) \
    get_os_is_zyx_isv8_osv16_isv2_index(                                                  \
        0, o, i, z, y, x,                                                                 \
        CAT(prefix, _SIZE_X),                                                             \
        CAT(prefix, _SIZE_Y),                                                             \
        CAT(prefix, _SIZE_Z),                                                             \
        CAT(prefix, _GROUPS_NUM),                                                         \
        CAT(prefix, _OFM_NUM),                                                            \
        CAT(prefix, _IFM_NUM),                                                            \
        CAT(prefix, _OFFSET)                                                              \
    )

inline uint get_os_is_zyx_isv_osv_index(uint o, uint i, uint z, uint y, uint x,
    uint x_size, uint y_size, uint z_size, uint i_size, uint o_size, uint osv_size, uint isv_size)
{
    const uint isv = i % isv_size;
    const uint osv = o % osv_size;
    const uint is = i / isv_size;
    const uint os = o / osv_size;

    const uint x_pitch = osv_size * isv_size;
    const uint y_pitch = x_pitch * x_size;
    const uint z_pitch = y_pitch * y_size;
    const uint is_pitch = z_pitch * z_size;
    const uint os_pitch = is_pitch * ((i_size + isv_size - 1) / isv_size);

    const uint output_offset =
        osv +
        isv * osv_size +
        x * x_pitch +
        y * y_pitch +
        z * z_pitch +
        is * is_pitch +
        os * os_pitch;

    return output_offset;
}

inline uint get_is_os_zyx_isv_osv_index(uint o, uint i, uint z, uint y, uint x,
    uint x_size, uint y_size, uint z_size, uint i_size, uint o_size, uint osv_size, uint isv_size)
{
    const uint isv = i % isv_size;
    const uint osv = o % osv_size;
    const uint is = i / isv_size;
    const uint os = o / osv_size;

    const uint x_pitch = osv_size * isv_size;
    const uint y_pitch = x_pitch * x_size;
    const uint z_pitch = y_pitch * y_size;
    const uint os_pitch = z_pitch * z_size;
    const uint is_pitch = os_pitch * ((o_size + osv_size - 1) / osv_size);

    const uint output_offset =
        osv +
        isv * osv_size +
        x * x_pitch +
        y * y_pitch +
        z * z_pitch +
        os * os_pitch +
        is * is_pitch;

    return output_offset;
}

inline uint get_os_is_zyx_osv_isv_index(uint o, uint i, uint z, uint y, uint x,
    uint x_size, uint y_size, uint z_size, uint i_size, uint o_size, uint osv_size, uint isv_size)
{
    const uint isv = i % isv_size;
    const uint osv = o % osv_size;
    const uint is = i / isv_size;
    const uint os = o / osv_size;

    const uint x_pitch = osv_size * isv_size;
    const uint y_pitch = x_pitch * x_size;
    const uint z_pitch = y_pitch * y_size;
    const uint is_pitch = z_pitch * z_size;
    const uint os_pitch = is_pitch * ((i_size + isv_size - 1) / isv_size);

    const uint output_offset =
        isv +
        osv * isv_size +
        x * x_pitch +
        y * y_pitch +
        z * z_pitch +
        is * is_pitch +
        os * os_pitch;

    return output_offset;
}

inline uint get_g_os_is_zyx_osv_isv_index(uint g, uint o, uint i, uint z, uint y, uint x,
    uint x_size, uint y_size, uint z_size, uint i_size, uint o_size, uint osv_size, uint isv_size)
{
    const uint isv = i % isv_size;
    const uint osv = o % osv_size;
    const uint is = i / isv_size;
    const uint os = o / osv_size;

    const uint x_pitch = osv_size * isv_size;
    const uint y_pitch = x_pitch * x_size;
    const uint z_pitch = y_pitch * y_size;
    const uint is_pitch = z_pitch * z_size;
    const uint os_pitch = is_pitch * ((i_size + isv_size - 1) / isv_size);
    const uint g_pitch = os_pitch * ((o_size + osv_size - 1) / osv_size);

    const uint output_offset =
        isv +
        osv * isv_size +
        x * x_pitch +
        y * y_pitch +
        z * z_pitch +
        is * is_pitch +
        os * os_pitch +
        g * g_pitch;

    return output_offset;
}

#define GET_FILTER_G_OS_IS_ZYX_OSV16_ISV16_INDEX(prefix, g, o, i, z, y, x)   \
    get_g_os_is_zyx_osv_isv_index(                                           \
        g, o, i, z, y, x,                                                    \
        CAT(prefix, _SIZE_X),                                                \
        CAT(prefix, _SIZE_Y),                                                \
        CAT(prefix, _SIZE_Z),                                                \
        CAT(prefix, _IFM_NUM),                                               \
        CAT(prefix, _OFM_NUM),                                               \
        16,                                                                  \
        16)

#define GET_FILTER_OS_IS_YX_OSV16_ISV16_INDEX(prefix, o, i, y, x) \
    get_os_is_zyx_osv_isv_index(                                  \
        o, i, 0, y, x,                                            \
        CAT(prefix, _SIZE_X),                                     \
        CAT(prefix, _SIZE_Y),                                     \
        1,                                                        \
        CAT(prefix, _IFM_NUM),                                    \
        CAT(prefix, _OFM_NUM),                                    \
        16,                                                       \
        16)

#define GET_FILTER_OS_IS_ZYX_OSV16_ISV16_INDEX(prefix, o, i, z, y, x)   \
    get_os_is_zyx_osv_isv_index(                                        \
        o, i, z, y, x,                                                  \
        CAT(prefix, _SIZE_X),                                           \
        CAT(prefix, _SIZE_Y),                                           \
        CAT(prefix, _SIZE_Z),                                           \
        CAT(prefix, _IFM_NUM),                                          \
        CAT(prefix, _OFM_NUM),                                          \
        16,                                                             \
        16)

#define GET_FILTER_OS_IS_ZYX_OSV32_ISV16_INDEX(prefix, o, i, z, y, x)   \
    get_os_is_zyx_osv_isv_index(                                        \
        o, i, z, y, x,                                                  \
        CAT(prefix, _SIZE_X),                                           \
        CAT(prefix, _SIZE_Y),                                           \
        CAT(prefix, _SIZE_Z),                                           \
        CAT(prefix, _IFM_NUM),                                          \
        CAT(prefix, _OFM_NUM),                                          \
        32,                                                             \
        16)

#define GET_FILTER_OS_IS_ZYX_OSV64_ISV16_INDEX(prefix, o, i, z, y, x)   \
    get_os_is_zyx_osv_isv_index(                                        \
        o, i, z, y, x,                                                  \
        CAT(prefix, _SIZE_X),                                           \
        CAT(prefix, _SIZE_Y),                                           \
        CAT(prefix, _SIZE_Z),                                           \
        CAT(prefix, _IFM_NUM),                                          \
        CAT(prefix, _OFM_NUM),                                          \
        64,                                                             \
        16)

#define GET_FILTER_G_OS_IS_YX_ISV8_OSV16_ISV2_INDEX(prefix, g, o, i, y, x, sub_group_size) \
    get_os_is_zyx_isv8_osv16_isv2_index(                                                   \
        g, o, i, 0, y, x,                                                                  \
        CAT(prefix, _SIZE_X),                                                              \
        CAT(prefix, _SIZE_Y),                                                              \
        CAT(prefix, _SIZE_Z),                                                              \
        CAT(prefix, _GROUPS_NUM),                                                          \
        CAT(prefix, _OFM_NUM),                                                             \
        CAT(prefix, _IFM_NUM),                                                             \
        CAT(prefix, _OFFSET)                                                               \
    )

#define GET_FILTER_G_OS_IS_ZYX_ISV8_OSV16_ISV2_INDEX(prefix, g, o, i, z, y, x, sub_group_size) \
    get_os_is_zyx_isv8_osv16_isv2_index(                                                       \
        g, o, i, z, y, x,                                                                      \
        CAT(prefix, _SIZE_X),                                                                  \
        CAT(prefix, _SIZE_Y),                                                                  \
        CAT(prefix, _SIZE_Z),                                                                  \
        CAT(prefix, _GROUPS_NUM),                                                              \
        CAT(prefix, _OFM_NUM),                                                                 \
        CAT(prefix, _IFM_NUM),                                                                 \
        CAT(prefix, _OFFSET)                                                                   \
    )

inline uint get_os_is_zyx_isv8_osv16_isv2_index(uint g, uint o, uint i,  uint z, uint y, uint x, uint x_size, uint y_size, uint z_size,
                                                      uint g_size, uint o_size, uint i_size, uint offset)
{
    const uint group_offset = g * o_size * i_size * z_size * y_size * x_size;
    const uint xyz_offset = (x + y * x_size + z * x_size * y_size)* 8*16*2;

    const uint i2_val = i % 2;
    const uint i2_slice = i / 2;
    const uint i8_v = i2_slice % 8;
    const uint i8_s = i2_slice / 8;

    const uint i2_offset = i2_val;
    const uint o_offset = (o % 16)*2 + (o / 16) * 16 * i_size * x_size * y_size * z_size;
    const uint i8_offset = 8*16*2* x_size*y_size*z_size * i8_s + 16*2*i8_v;

    const size_t idx = offset + group_offset + xyz_offset + i2_offset + i8_offset + o_offset;

    return idx;
}

inline uint get_os_zyxi_osv16_index(uint o, uint i, uint z, uint y, uint x, uint i_size, uint o_size, uint x_size, uint y_size, uint z_size)
{
    const size_t idx = o%16 + (o / 16)*i_size*x_size*y_size*z_size*16 +
                       16 *(i+ x*i_size + y*i_size*x_size + z*i_size*x_size*y_size);
    return idx;
}

#define GET_FILTER_OS_ZYXI_OSV16(prefix, o, i, z, y, x) \
    get_os_zyxi_osv16_index(                            \
        o, i, z, y, x, CAT(prefix, _IFM_NUM),           \
        CAT(prefix, _OFM_NUM),                          \
        CAT(prefix, _SIZE_X),                           \
        CAT(prefix, _SIZE_Y),                           \
        CAT(prefix, _SIZE_Z))

#define GET_FILTER_GOIYX(prefix, g, o, i, y, x) \
    CAT(prefix, _OFFSET) +                      \
    (x)*CAT(prefix, _X_PITCH) +                 \
    (y)*CAT(prefix, _Y_PITCH) +                 \
    (i)*CAT(prefix, _IFM_PITCH) +               \
    (o)*CAT(prefix, _OFM_PITCH) +               \
    (g)*CAT(prefix, _GROUPS_PITCH)

#define GET_FILTER_GIOYX(prefix, g, o, i, y, x) \
    CAT(prefix, _OFFSET) +                      \
    (x)*CAT(prefix, _X_PITCH) +                 \
    (y)*CAT(prefix, _Y_PITCH) +                 \
    (i)*CAT(prefix, _IFM_PITCH) +               \
    (o)*CAT(prefix, _OFM_PITCH) +               \
    (g)*CAT(prefix, _GROUPS_PITCH)

#define GET_FILTER_GIOYX_SAFE(prefix, g, o, i, y, x)        \
    CAT(prefix, _OFFSET) +                                  \
    (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) +     \
    (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) +     \
    (i % CAT(prefix, _IFM_NUM))*CAT(prefix, _IFM_PITCH) +   \
    (o % CAT(prefix, _OFM_NUM))*CAT(prefix, _OFM_PITCH) +   \
    (g % CAT(prefix, _GROUPS_NUM))*CAT(prefix, _GROUPS_PITCH)

#define GET_FILTER_GOIYX_SAFE(prefix, g, o, i, y, x)        \
    CAT(prefix, _OFFSET) +                                  \
    (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) +     \
    (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) +     \
    (i % CAT(prefix, _IFM_NUM))*CAT(prefix, _IFM_PITCH) +   \
    (o % CAT(prefix, _OFM_NUM))*CAT(prefix, _OFM_PITCH) +   \
    (g % CAT(prefix, _GROUPS_NUM))*CAT(prefix, _GROUPS_PITCH)

#define GET_FILTER_INDEX(prefix, g, o, i, y, x) GET_FILTER_GOIYX(prefix, g, o, i, y, x)

#define GET_FILTER_INDEX_SAFE(prefix, g, o, i, y, x) GET_FILTER_GOIYX_SAFE(prefix, g, o, i, y, x)

#define GET_FILTER_GOIZYX(prefix, g, o, i, z, y, x) \
    CAT(prefix, _OFFSET) +                          \
    (x)*CAT(prefix, _X_PITCH) +                     \
    (y)*CAT(prefix, _Y_PITCH) +                     \
    (z)*CAT(prefix, _Z_PITCH) +                     \
    (i)*CAT(prefix, _IFM_PITCH) +                   \
    (o)*CAT(prefix, _OFM_PITCH) +                   \
    (g)*CAT(prefix, _GROUPS_PITCH)

#define GET_FILTER_GOIZYX_SAFE(prefix, g, o, i, z, y, x)    \
    CAT(prefix, _OFFSET) +                                  \
    (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) +     \
    (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) +     \
    (z % CAT(prefix, _SIZE_Z ))*CAT(prefix, _Z_PITCH) +     \
    (i % CAT(prefix, _IFM_NUM))*CAT(prefix, _IFM_PITCH) +   \
    (o % CAT(prefix, _OFM_NUM))*CAT(prefix, _OFM_PITCH) +   \
    (g % CAT(prefix, _GROUPS_NUM))*CAT(prefix, _GROUPS_PITCH)

#define GET_FILTER_GIOZYX(prefix, g, o, i, z, y, x) \
    CAT(prefix, _OFFSET) +                          \
    (x)*CAT(prefix, _X_PITCH) +                     \
    (y)*CAT(prefix, _Y_PITCH) +                     \
    (z)*CAT(prefix, _Z_PITCH) +                     \
    (i)*CAT(prefix, _IFM_PITCH) +                   \
    (o)*CAT(prefix, _OFM_PITCH) +                   \
    (g)*CAT(prefix, _GROUPS_PITCH)

#define GET_FILTER_GIOZYX_SAFE(prefix, g, o, i, z, y, x)    \
    CAT(prefix, _OFFSET) +                                  \
    (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) +     \
    (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) +     \
    (z % CAT(prefix, _SIZE_Z ))*CAT(prefix, _Z_PITCH) +     \
    (i % CAT(prefix, _IFM_NUM))*CAT(prefix, _IFM_PITCH) +   \
    (o % CAT(prefix, _OFM_NUM))*CAT(prefix, _OFM_PITCH) +   \
    (g % CAT(prefix, _GROUPS_NUM))*CAT(prefix, _GROUPS_PITCH)

#define GET_FILTER_INDEX_5D(prefix, g, o, i, z, y, x) GET_FILTER_GOIZYX(prefix, g, o, i, z, y, x)

#define GET_FILTER_INDEX_5D_SAFE(prefix, g, o, i, z, y, x) GET_FILTER_GOIZYX_SAFE(prefix, g, o, i, z, y, x)

#define GET_FILTER_OS_IYX_OSV_INDEX(prefix, o, i, y, x, sub_group_size) \
    CAT(prefix, _OFFSET) +                                               \
    ((o) % (sub_group_size)) +                                           \
    (sub_group_size)*(                                                   \
        (x)*CAT(prefix, _X_PITCH) +                                      \
        (y)*CAT(prefix, _Y_PITCH) +                                      \
        (i)*CAT(prefix, _IFM_PITCH) +                                    \
        ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH)                 \
    )

#define GET_FILTER_OS_IYX_OSV_INDEX_INT4_PACKED(prefix, o, i, y, x, sub_group_size) \
    CAT(prefix, _OFFSET) +                                               \
    ((o) % (sub_group_size)) +                                           \
    (sub_group_size)*(                                                   \
        (x)*CAT(prefix, _X_PITCH) +                                      \
        (y)*CAT(prefix, _Y_PITCH) +                                      \
        (i)*CAT(prefix, _IFM_PITCH) +                                    \
        ((o) / (sub_group_size))*(CAT(prefix, _OFM_PITCH)/2)             \
    )

#define GET_FILTER_OS_IS_YX_OSV_ISV_INDEX_INT4_PACKED(prefix, o, i, y, x, sub_group_size) \
    CAT(prefix, _OFFSET) +                                               \
    ((o) % (sub_group_size)) +                                           \
    (sub_group_size)*(                                                   \
        (x)*CAT(prefix, _X_PITCH) +                                      \
        (y)*CAT(prefix, _Y_PITCH) +                                      \
        (i)*CAT(prefix, _IFM_PITCH) +                                    \
        ((o) / (sub_group_size))*(CAT(prefix, _OFM_PITCH)/2)                 \
    )

#define GET_FILTER_OS_IYX_OSV_ROTATE_180_INDEX(prefix, o, i, y, x, sub_group_size) \
    CAT(prefix, _OFFSET) +                                                          \
    ((o) % (sub_group_size)) +                                                      \
    (sub_group_size)*(                                                              \
        (CAT(prefix, _SIZE_X ) - x - 1)*CAT(prefix, _X_PITCH) +                     \
        (CAT(prefix, _SIZE_Y ) - y - 1)*CAT(prefix, _Y_PITCH) +                     \
        (i)*CAT(prefix, _IFM_PITCH) +                                               \
        ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH)                            \
    )

inline uint get_gi_yxs_os_yxsv2_osv_index(uint g, uint o, uint i, uint y, uint x, uint x_size, uint g_pitch, uint i_pitch,
                                                uint y_pitch, uint x_pitch, uint offset, uint sub_group_size)
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
    size_t idx = offset + aligned_height*aligned_ofm_line + in_line;

    idx += g * g_pitch;

    return idx;
}

#define GET_FILTER_I_YXS_OS_YXSV2_OSV_INDEX(prefix, o, i, y, x, sub_group_size) \
    get_gi_yxs_os_yxsv2_osv_index(                                              \
        0, o, i, y, x,                                                          \
        CAT(prefix, _SIZE_X ),                                                  \
        CAT(prefix, _GROUPS_PITCH),                                             \
        CAT(prefix, _IFM_PITCH),                                                \
        CAT(prefix, _Y_PITCH),                                                  \
        CAT(prefix, _X_PITCH),                                                  \
        CAT(prefix, _OFFSET),                                                   \
        sub_group_size)

inline uint get_giy_xs_os_xsv2_osv_index(uint g, uint o, uint i, uint y, uint x, uint x_size, uint g_pitch,
                                               uint i_pitch, uint y_pitch, uint x_pitch, uint offset, uint sub_group_size)
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
    size_t idx = offset + aligned_height*aligned_ofm_line + in_line;

    idx += g * g_pitch;

    return idx;
}

#define GET_FILTER_IY_XS_OS_XSV2_OSV_INDEX(prefix, o, i, y, x, sub_group_size)  \
    get_giy_xs_os_xsv2_osv_index(                                               \
        0, o, i, y, x,                                                          \
        CAT(prefix, _SIZE_X ),                                                  \
        CAT(prefix, _GROUPS_PITCH),                                             \
        CAT(prefix, _IFM_PITCH),                                                \
        CAT(prefix, _Y_PITCH),                                                  \
        CAT(prefix, _X_PITCH),                                                  \
        CAT(prefix, _OFFSET),                                                   \
        sub_group_size)

inline uint get_is_os_zyx_isa8_osv8_isv2_index(uint o, uint i, uint z, uint y, uint x, uint size_x,
                                              uint size_y, uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
    const uint isv2_idx = i % 2;
    const uint osv_idx = o % 8;
    const uint isv1_idx = (i / 2) % 8;
    const uint is_idx = i / 16;
    const uint os_idx = o / 8;

    const uint of_8_aligned = ((size_ofm + 7) / 8);

    size_t idx = offset +
                 isv2_idx +
                 osv_idx * 2 +
                 isv1_idx * 8 * 2 +
                 x * 8 * 8 * 2 +
                 y * size_x * 8 * 8 * 2 +
                 z * size_y * size_x * 8 * 8 * 2 +
                 os_idx * size_z * size_y * size_x * 8 * 8 * 2 +
                 is_idx * of_8_aligned * size_z * size_y * size_x * 8 * 8 * 2;

    return idx;
}

inline uint get_g_os_is_zyx_isa_osv_isv_index(uint g, uint o, uint i, uint z, uint y, uint x,
                                              uint size_x, uint size_y, uint size_z, uint size_ifm, uint size_ofm, uint offset,
                                              uint isa, uint osv, uint isv)
{
    const uint isv2_idx = i % isv;
    const uint osv_idx = o % osv;
    const uint isv1_idx = (i / isv) % isa;
    const uint is_idx = i / (isa * isv);
    const uint os_idx = o / osv;

    const uint if_aligned = ((size_ifm + (isa * isv) - 1) / (isa * isv));
    const uint of_aligned = ((size_ofm + (osv - 1)) / osv);

    size_t idx = offset +
                 isv2_idx +
                 osv_idx * isv +
                 isv1_idx * osv * isv +
                 x * isa * osv * isv +
                 y * size_x * isa * osv * isv +
                 z * size_y * size_x * isa * osv * isv +
                 is_idx * size_z * size_y * size_x * isa * osv * isv +
                 os_idx * if_aligned * size_z * size_y * size_x * isa * osv * isv +
                 g * of_aligned * if_aligned * size_z * size_y * size_x * isa * osv * isv;

    return idx;
}

#define GET_FILTER_G_OS_IS_ZYX_ISA_OSV_ISV_INDEX(prefix, g, o, i, z, y, x, isa, osv, isv) \
    get_g_os_is_zyx_isa_osv_isv_index(                                                    \
        g, o, i, z, y, x,                                                                 \
        CAT(prefix, _SIZE_X),                                                             \
        CAT(prefix, _SIZE_Y),                                                             \
        CAT(prefix, _SIZE_Z),                                                             \
        CAT(prefix, _IFM_NUM),                                                            \
        CAT(prefix, _OFM_NUM),                                                            \
        CAT(prefix, _OFFSET),                                                             \
        isa, osv, isv)

inline uint get_g_os_is_yx_isa8_osv8_isv4_index(uint g, uint o, uint i, uint y, uint x, uint size_x,
                                                uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
    const uint isv2_idx = i % 4;
    const uint osv_idx = o % 8;
    const uint isv1_idx = (i / 4) % 8;
    const uint is_idx = i / 32;
    const uint os_idx = o / 8;

    const uint if_32_aligned = ((size_ifm + 31) / 32);
    const uint of_8_aligned = ((size_ofm + 7) / 8);

    size_t idx = offset +
                 isv2_idx +
                 osv_idx * 4 +
                 isv1_idx * 8 * 4 +
                 x * 8 * 8 * 4 +
                 y * size_x * 8 * 8 * 4 +
                 is_idx * size_y * size_x * 4 * 8 * 8 +
                 os_idx * if_32_aligned * size_y * size_x * 4 * 8 * 8 +
                 g * of_8_aligned * if_32_aligned * size_y * size_x * 4 * 8 * 8;

    return idx;
}

#define GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_INDEX(prefix, o, i, y, x)      \
    get_g_os_is_yx_isa8_osv8_isv4_index(                                  \
        0, o, i, y, x, CAT(prefix, _SIZE_X ),                             \
        CAT(prefix, _SIZE_Y),                                             \
        CAT(prefix, _IFM_NUM),                                            \
        CAT(prefix, _OFM_NUM),                                            \
        CAT(prefix, _OFFSET))

inline uint get_is_os_yx_isa8_osv8_isv2_index(uint o, uint i, uint y, uint x, uint size_x,
                                              uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
	const uint isv2_idx = i % 2;
	const uint osv_idx = o % 8;
	const uint isv1_idx = (i / 2) % 8;
	const uint is_idx = i / 16;
	const uint os_idx = o / 8;

    const uint of_8_aligned = ((size_ofm + 7) / 8);

	size_t idx = offset +
                 isv2_idx +
                 osv_idx * 2 +
                 isv1_idx * 8 * 2 +
                 x * 8 * 8 * 2 +
                 y * size_x * 8 * 8 * 2 +
                 os_idx * size_y * size_x * 2 * 8 * 8 +
                 is_idx * of_8_aligned * size_y * size_x * 2 * 8 * 8;

    return idx;
}

inline uint get_is_os_yx_isa8_osv8_isv4_index(uint o, uint i, uint y, uint x, uint size_x,
                                              uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
	const uint isv2_idx = i % 4;
	const uint osv_idx = o % 8;
	const uint isv1_idx = (i / 4) % 8;
	const uint is_idx = i / 32;
	const uint os_idx = o / 8;

    const uint of_8_aligned = ((size_ofm + 7) / 8);

	size_t idx = offset +
                 isv2_idx +
                 osv_idx * 4 +
                 isv1_idx * 8 * 4 +
                 x * 8 * 8 * 4 +
                 y * size_x * 8 * 8 * 4 +
                 os_idx * size_y * size_x * 4 * 8 * 8 +
                 is_idx * of_8_aligned * size_y * size_x * 4 * 8 * 8;

    return idx;
}

inline uint get_is_os_yx_osa8_isv16_osv4_index(uint o, uint i, uint y, uint x, uint size_x,
                                               uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
	const uint osv2_idx = o % 4;
	const uint isv_idx = i % 16;
	const uint osv1_idx = (o / 4) % 8;
	const uint os_idx = o / 32;
	const uint is_idx = i / 16;

    const uint of_32_aligned = ((size_ofm + 31) / 32);

	size_t idx = offset +
                 osv2_idx +
                 isv_idx * 4 +
                 osv1_idx * 16 * 4 +
                 x * 8 * 16 * 4 +
                 y * size_x * 8 * 16 * 4 +
                 os_idx * size_y * size_x * 4 * 16 * 8 +
                 is_idx * of_32_aligned * size_y * size_x * 4 * 16 * 8;

    return idx;
}

inline uint get_os_is_zyx_isa8_osv8_isv4_index(uint o, uint i, uint z, uint y, uint x,
                                                     uint size_x, uint size_y, uint size_z,
                                                     uint size_ifm, uint size_ofm, uint offset)
{
    const uint ifm_slices = (size_ifm + 31)/32;
    const uint isv2_idx = i % 4;
    const uint osv_idx = o % 8;
    const uint isv1_idx = (i / 4) % 8;
    const uint is_idx = i / 32;
    const uint os_idx = o / 8;

    size_t idx = offset + isv2_idx + 4 * (osv_idx + 8 * isv1_idx);
    idx += x * 4 * 8 * 8;
    idx += y * size_x * 4 * 8 * 8;
    idx += z * size_y * size_x * 4 * 8 * 8;
    idx += is_idx * size_z * size_y * size_x * 4 * 8 * 8;
    idx += os_idx * ifm_slices * size_z * size_y * size_x * 4 * 8 * 8;

    return idx;
}

#define GET_FILTER_OS_IS_ZYX_ISA8_OSV8_ISV4_INDEX(prefix, o, i, z, y, x) \
    get_os_is_zyx_isa8_osv8_isv4_index(                                  \
        o, i, z, y, x,                                                   \
        CAT(prefix, _SIZE_X ),                                           \
        CAT(prefix, _SIZE_Y),                                            \
        CAT(prefix, _SIZE_Z),                                            \
        CAT(prefix, _IFM_NUM),                                           \
        CAT(prefix, _OFM_NUM),                                           \
        CAT(prefix, _OFFSET))


inline uint get_os_is_yx_isa8_osv16_isv4_index(uint o, uint i, uint y, uint x, uint size_x, uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
    const uint f_32_aligned = ((size_ifm + 31)/32) * 32;
    const uint isv2_idx = i % 4;
    const uint osv_idx = o % 16;
    const uint isv1_idx = (i / 4) % 8;
    const uint is_idx = i / 32;
    const uint os_idx = o / 16;

    size_t idx = offset + isv2_idx + 4 * (osv_idx + 16 * isv1_idx);
    idx += x * 4 * 8 * 16;
    idx += y * size_x * 4 * 8 * 16;
    idx += is_idx * size_y * size_x * 4 * 8 * 16;
    idx += os_idx * (f_32_aligned/32) * size_y * size_x * 4 * 8 * 16;

    return idx;
}

#define GET_FILTER_OS_IS_YX_ISA8_OSV16_ISV4_INDEX(prefix, o, i, y, x) \
    get_os_is_yx_isa8_osv16_isv4_index(                               \
        o, i, y, x, CAT(prefix, _SIZE_X ),                            \
        CAT(prefix, _SIZE_Y),                                         \
        CAT(prefix, _IFM_NUM),                                        \
        CAT(prefix, _OFM_NUM),                                        \
        CAT(prefix, _OFFSET))

inline uint get_os_is_zyx_isa8_osv16_isv4_index(uint o, uint i, uint z, uint y, uint x,
                                                      uint size_x, uint size_y, uint size_z,
                                                      uint size_ifm, uint size_ofm, uint offset)
{
    const uint ifm_slices = (size_ifm + 31)/32;
    const uint isv2_idx = i % 4;
    const uint osv_idx = o % 16;
    const uint isv1_idx = (i / 4) % 8;
    const uint is_idx = i / 32;
    const uint os_idx = o / 16;

    size_t idx = offset + isv2_idx + 4 * (osv_idx + 16 * isv1_idx);
    idx += x * 4 * 8 * 16;
    idx += y * size_x * 4 * 8 * 16;
    idx += z * size_y * size_x * 4 * 8 * 16;
    idx += is_idx * size_z * size_y * size_x * 4 * 8 * 16;
    idx += os_idx * ifm_slices * size_z * size_y * size_x * 4 * 8 * 16;

    return idx;
}

#define GET_FILTER_OS_IS_ZYX_ISA8_OSV16_ISV4_INDEX(prefix, o, i, z, y, x) \
    get_os_is_zyx_isa8_osv16_isv4_index(                                  \
        o, i, z, y, x,                                                    \
        CAT(prefix, _SIZE_X ),                                            \
        CAT(prefix, _SIZE_Y),                                             \
        CAT(prefix, _SIZE_Z),                                             \
        CAT(prefix, _IFM_NUM),                                            \
        CAT(prefix, _OFM_NUM),                                            \
        CAT(prefix, _OFFSET))

inline uint get_os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4_index(uint o, uint i, uint y, uint x, uint size_x, uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
    const uint o_swizzled = (o % 4) * 8 + ((o % 32) / 4) + (o / 32) * 32;
    const uint isv_idx = i % 4;
    const uint isa_idx = (i / 4) % 8;
    const uint is_idx = (i / 32);
    const uint osv_idx = o_swizzled % 8;
    const uint osa_idx = (o_swizzled / 8) % 4;
    const uint os_idx = (o / 32);

    const uint f_32_aligned = ((size_ifm + 31)/32);

    size_t idx = offset +
                 isv_idx +
                 osv_idx * 4 +
                 isa_idx * 8 * 4 +
                 osa_idx * 8 * 32 +
                 x * 32 * 32 +
                 y * size_x * 32 * 32 +
                 is_idx * 32 * 32 * size_x * size_y +
                 os_idx * 32 * 32 * f_32_aligned * size_x * size_y;

    return idx;
}

inline uint get_os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4_index(uint o, uint i, uint z, uint y, uint x,
                                                                        uint size_x, uint size_y, uint size_z,
                                                                        uint size_ifm, uint size_ofm, uint offset)
{
    const uint o_swizzled = (o % 4) * 8 + ((o % 32) / 4) + (o / 32) * 32;
    const uint isv_idx = i % 4;
    const uint isa_idx = (i / 4) % 8;
    const uint is_idx = (i / 32);
    const uint osv_idx = o_swizzled % 8;
    const uint osa_idx = (o_swizzled / 8) % 4;
    const uint os_idx = (o / 32);

    const uint f_32_aligned = ((size_ifm + 31)/32);

    size_t idx = offset +
                 isv_idx +
                 osv_idx * 4 +
                 isa_idx * 8 * 4 +
                 osa_idx * 8 * 32 +
                 x * 32 * 32 +
                 y * size_x * 32 * 32 +
                 z * size_x * size_y * 32 * 32 +
                 is_idx * 32 * 32 * size_x * size_y * size_z +
                 os_idx * 32 * 32 * f_32_aligned * size_x * size_y * size_z;

    return idx;
}

inline uint get_g_is_os_yx_osa4_isa8_osv8_isv4(uint g, uint o, uint i, uint z, uint y, uint x,
                                               uint size_x, uint size_y, uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
    const uint isv_idx = i % 4;
    const uint isa_idx = (i / 4) % 8;
    const uint is_idx = (i / 32);
    const uint osv_idx = o % 8;
    const uint osa_idx = (o / 8) % 4;
    const uint os_idx = (o / 32);

    const uint ifm_32_aligned = ((size_ifm + 31) / 32);
    const uint ofm_32_aligned = ((size_ofm + 31) / 32);

    size_t idx = offset +
                 isv_idx +
                 osv_idx * 4 +
                 isa_idx * 8 * 4 +
                 osa_idx * 8 * 32 +
                 x * 32 * 32 +
                 y * size_x * 32 * 32 +
                 z * size_y * size_x * 32 * 32 +
                 os_idx * 32 * 32 * size_x * size_y * size_z +
                 is_idx * 32 * 32 * ofm_32_aligned * size_x * size_y * size_z +
                 g * 32 * 32 * ifm_32_aligned * ofm_32_aligned * size_x * size_y * size_z;

    return idx;
}

inline uint get_g_os_is_yx_osa4_isa8_osv8_isv4(uint g, uint o, uint i, uint z, uint y, uint x,
                                                     uint size_x, uint size_y, uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
    const uint isv_idx = i % 4;
    const uint isa_idx = (i / 4) % 8;
    const uint is_idx = (i / 32);
    const uint osv_idx = o % 8;
    const uint osa_idx = (o / 8) % 4;
    const uint os_idx = (o / 32);

    const uint ifm_32_aligned = ((size_ifm + 31)/32);
    const uint ofm_32_aligned = ((size_ofm + 31)/32);

    size_t idx = offset +
                 isv_idx +
                 osv_idx * 4 +
                 isa_idx * 8 * 4 +
                 osa_idx * 8 * 32 +
                 x * 32 * 32 +
                 y * size_x * 32 * 32 +
                 z * size_y * size_x * 32 * 32 +
                 is_idx * 32 * 32 * size_x * size_y * size_z +
                 os_idx * 32 * 32 * ifm_32_aligned * size_x * size_y * size_z +
                 g * 32 * 32 * ifm_32_aligned * ofm_32_aligned * size_x * size_y * size_z;

    return idx;
}

inline uint get_g_os_is_yx_osa4_isa8_osv8_isv2(uint g, uint o, uint i, uint z, uint y, uint x,
                                               uint size_x, uint size_y,  uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
    const uint isv_idx = i % 2;
    const uint isa_idx = (i / 2) % 8;
    const uint is_idx = (i / 16);
    const uint osv_idx = o % 8;
    const uint osa_idx = (o / 8) % 4;
    const uint os_idx = (o / 32);

    const uint ifm_16_aligned = ((size_ifm + 15)/16);
    const uint ofm_32_aligned = ((size_ofm + 31)/32);

    size_t idx = offset +
                 isv_idx +
                 osv_idx * 2 +
                 isa_idx * 8 * 2 +
                 osa_idx * 8 * 16 +
                 x * 32 * 16 +
                 y * size_x * 32 * 16 +
                 z * size_y * size_x * 32 * 16 +
                 is_idx * 32 * 16 * size_x * size_y * size_z +
                 os_idx * 32 * 16 * ifm_16_aligned * size_x * size_y * size_z +
                 g * 32 * 16 * ifm_16_aligned * ofm_32_aligned * size_x * size_y * size_z;

    return idx;
}

inline uint get_g_os_is_yx_osa2_isa8_osv8_isv2(uint g, uint o, uint i, uint z, uint y, uint x,
                                               uint size_x, uint size_y,  uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
    const uint isv_idx = i % 2;
    const uint isa_idx = (i / 2) % 8;
    const uint is_idx = (i / 16);
    const uint osv_idx = o % 8;
    const uint osa_idx = (o / 8) % 2;
    const uint os_idx = (o / 16);

    const uint ifm_16_aligned = ((size_ifm + 15)/16);
    const uint ofm_16_aligned = ((size_ofm + 15)/16);

    size_t idx = offset +
                 isv_idx +
                 osv_idx * 2 +
                 isa_idx * 8 * 2 +
                 osa_idx * 8 * 16 +
                 x * 16 * 16 +
                 y * size_x * 16 * 16 +
                 z * size_y * size_x * 16 * 16 +
                 is_idx * 16 * 16 * size_x * size_y * size_z +
                 os_idx * 16 * 16 * ifm_16_aligned * size_x * size_y * size_z +
                 g * 16 * 16 * ifm_16_aligned * ofm_16_aligned * size_x * size_y * size_z;

    return idx;
}

inline uint get_g_is_os_yx_isa2_osa8_isv8_osv2(uint g, uint o, uint i, uint z, uint y, uint x,
                                               uint size_x, uint size_y,  uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
    return get_g_os_is_yx_osa2_isa8_osv8_isv2(g, i, o, z, y, x, size_x, size_y, size_z, size_ofm, size_ifm, offset);
}

inline uint get_g_is_os_yx_isa4_osa8_isv8_osv4(uint g, uint o, uint i, uint z, uint y, uint x,
                                               uint size_x, uint size_y,  uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
    return get_g_os_is_yx_osa4_isa8_osv8_isv4(g, i, o, z, y, x, size_x, size_y, size_z, size_ofm, size_ifm, offset);
}

#define GET_FILTER_OS_IS_YX_OSA4_ISA8_OSV8_ISV4_INDEX(prefix, o, i, y, x) \
    get_g_os_is_yx_osa4_isa8_osv8_isv4(                                   \
        0, o, i, 0, y, x,                                                 \
        CAT(prefix, _SIZE_X),                                             \
        CAT(prefix, _SIZE_Y),                                             \
        1,                                                                \
        CAT(prefix, _IFM_NUM),                                            \
        CAT(prefix, _OFM_NUM),                                            \
        CAT(prefix, _OFFSET))

#define GET_FILTER_OS_IS_ZYX_OSA4_ISA8_OSV8_ISV4_INDEX(prefix, o, i, z, y, x) \
    get_g_os_is_yx_osa4_isa8_osv8_isv4(                                       \
        0, o, i, z, y, x,                                                     \
        CAT(prefix, _SIZE_X),                                                 \
        CAT(prefix, _SIZE_Y),                                                 \
        CAT(prefix, _SIZE_Z),                                                 \
        CAT(prefix, _IFM_NUM),                                                \
        CAT(prefix, _OFM_NUM),                                                \
        CAT(prefix, _OFFSET))

#define GET_FILTER_OS_IS_YX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(prefix, o, i, y, x) \
    get_os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4_index(                               \
        o, i, y, x,                                                                     \
        CAT(prefix, _SIZE_X),                                                           \
        CAT(prefix, _SIZE_Y),                                                           \
        CAT(prefix, _IFM_NUM),                                                          \
        CAT(prefix, _OFM_NUM),                                                          \
        CAT(prefix, _OFFSET))

#define GET_FILTER_OS_IS_ZYX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(prefix, o, i, z, y, x) \
    get_os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4_index(                                  \
        o, i, z, y, x,                                                                      \
        CAT(prefix, _SIZE_X),                                                               \
        CAT(prefix, _SIZE_Y),                                                               \
        CAT(prefix, _SIZE_Z),                                                               \
        CAT(prefix, _IFM_NUM),                                                              \
        CAT(prefix, _OFM_NUM),                                                              \
        CAT(prefix, _OFFSET))

inline uint get_is_o32_yx_isv32_swizzled_by_4_index(uint o, uint i, uint y, uint x, uint i_size, uint o_size, uint x_size, uint y_size)
{
    const uint o_aligned_to_32 = ((o_size + 31) / 32) * 32;
    const uint o_swizzled = (o % 4) * 8 + ((o % 32) / 4) + (o / 32) * 32;
    const uint i_aligned_to_32 = ((i_size + 31) / 32) * 32;
    const uint i_val = i % 32;
    const uint i_slice = i / 32;
    const size_t idx = i_val + 32* (x + x_size * (y + y_size * (o_swizzled + o_aligned_to_32 * i_slice) ) );
    return idx;
}

#define GET_FILTER_G_OS_IS_YX_OSV16_ISV4_INDEX(prefix, g, o, i, y, x) \
    get_g_os_is_yx_osv_isv(                                           \
        g, o, i, y, x,                                                \
        CAT(prefix, _IFM_NUM),                                        \
        CAT(prefix, _OFM_NUM),                                        \
        CAT(prefix, _SIZE_X),                                         \
        CAT(prefix, _SIZE_Y), 16, 4)

inline uint get_g_os_is_yx_osv_isv(uint g, uint o, uint i, uint y, uint x,
                                          uint i_size,
                                          uint o_size,
                                          uint x_size,
                                          uint y_size,
                                          uint osv_size,
                                          uint isv_size)
{
    return get_g_os_is_zyx_osv_isv_index(g, o, i, 0, y, x,
        x_size, y_size, 1, i_size, o_size, osv_size, isv_size);
}

#define GET_FILTER_OS_IS_YX_OSV8_ISV4_INDEX(prefix, o, i, y, x) \
    get_g_os_is_yx_osv_isv(                                     \
        0, o, i, y, x,                                          \
        CAT(prefix, _IFM_NUM),                                  \
        CAT(prefix, _OFM_NUM),                                  \
        CAT(prefix, _SIZE_X),                                   \
        CAT(prefix, _SIZE_Y), 8, 4)

#define GET_FILTER_OS_IS_YX_OSV16_ISV4_INDEX(prefix, o, i, y, x) \
    get_g_os_is_yx_osv_isv(                                      \
        0, o, i, y, x,                                           \
        CAT(prefix, _IFM_NUM),                                   \
        CAT(prefix, _OFM_NUM),                                   \
        CAT(prefix, _SIZE_X),                                    \
        CAT(prefix, _SIZE_Y), 16, 4)

#define GET_FILTER_OS_IS_YX_OSV32_ISV4_INDEX(prefix, o, i, y, x) \
    get_g_os_is_yx_osv_isv(                                      \
        0, o, i, y, x,                                           \
        CAT(prefix, _IFM_NUM),                                   \
        CAT(prefix, _OFM_NUM),                                   \
        CAT(prefix, _SIZE_X),                                    \
        CAT(prefix, _SIZE_Y), 32, 4)

#define GET_FILTER_OS_IS_ZYX_OSV32_ISV4_INDEX(prefix, o, i, z, y, x)    \
    get_os_is_zyx_osv_isv_index(                                        \
        o, i, z, y, x,                                                  \
        CAT(prefix, _SIZE_X),                                           \
        CAT(prefix, _SIZE_Y),                                           \
        CAT(prefix, _SIZE_Z),                                           \
        CAT(prefix, _IFM_NUM),                                          \
        CAT(prefix, _OFM_NUM),                                          \
        32,                                                             \
        4)

#define GET_FILTER_OS_IS_YX_OSV32_ISV4_SWIZZLED_BY_2_INDEX(prefix, o, i, y, x) \
    get_os_is_yx_osv32_isv4_swizzled_by_2(                                     \
        o, i, y, x,                                                            \
        CAT(prefix, _OFM_NUM),                                                 \
        CAT(prefix, _IFM_NUM),                                                 \
        CAT(prefix, _SIZE_Y),                                                  \
        CAT(prefix, _SIZE_X))

inline uint get_os_is_yx_osv32_isv4_swizzled_by_2(uint o, uint i, uint y, uint x,
                                                        uint o_size,
                                                        uint i_size,
                                                        uint y_size,
                                                        uint x_size)
{
    const uint osv = 32;
    const uint os = o / osv;
    const uint ofm_block = (o % osv) % 2;
    const uint ofm_in_block = (o % osv) / 2;

    const uint tile = 4;
    const uint ifm_aligned = ((i_size + tile - 1) / tile) * tile;
    const uint ifm_tile = i / tile;
    const uint id = i - ifm_tile * tile;

    uint idx = os * ifm_aligned * y_size * x_size * osv
               + ifm_tile * y_size * x_size * osv * tile
               + y * x_size * osv * tile
               + x * osv * tile
               + ofm_block * 16 * tile
               + ofm_in_block * tile
               + id;

    return idx;
}

inline uint get_os_is_osv32_isv32_swizzled_by_4_index(uint o, uint i, uint y, uint x, uint size_x, uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
    const uint size_ifm_a = ((size_ifm + 31)/32) * 32;

    const uint o_hi = o / 32;
    const uint o_lo = o % 32;
    const uint i_hi = i / 32;
    const uint i_lo = i % 32;

    const uint o_lo1 = o_lo % 4;
    const uint o_lo2 = (o_lo / 4) % 8;

    const uint i_lo1 = i_lo % 4;
    const uint i_lo2 = i_lo / 4;

    const uint idx_in_group = o_lo2 * 4 + o_lo1 * (32 * 8) + i_lo2 * 32 + i_lo1;
    const uint group_idx = o_hi * (size_ifm_a / 32) + i_hi;

    return group_idx * (32 * 32) + idx_in_group;
}

inline uint get_os_i_yxs_osv_yxsv4_index(uint o, uint i, uint y, uint x, uint i_size, uint size_x, uint size_y, uint osv) {
    const uint yxsv = 4;
    uint yx = y * size_x + x;
    uint yx_size_aligned = (size_x * size_y + yxsv - 1) / yxsv * yxsv;
    uint os_index = o / osv;
    uint yxs_index = yx / yxsv;
    uint osv_index = o % osv;
    uint yxsv_index = yx % yxsv;

    uint index = 0;
    index += yxsv_index;
    index += osv_index * yxsv;
    index += yxs_index * yxsv * osv;
    index += i * osv * yx_size_aligned;
    index += os_index * osv * yx_size_aligned * i_size;
    return index;
}

#define GET_FILTER_G_OS_IYX_OSV16(prefix, g, o, i, y, x, sub_group_size) \
    CAT(prefix, _OFFSET) +                                               \
    (g * CAT(prefix, _GROUPS_PITCH)) +                                   \
    ((o) % (sub_group_size)) +                                           \
    (sub_group_size)*(                                                   \
        (x)*CAT(prefix, _X_PITCH) +                                      \
        (y)*CAT(prefix, _Y_PITCH) +                                      \
        (i)*CAT(prefix, _IFM_PITCH) +                                    \
        ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH)                 \
    )

#define GET_FILTER_OS_IYX_OSV16(prefix, o, i, y, x, sub_group_size) GET_FILTER_G_OS_IYX_OSV16(prefix, 0, o, i, y, x, sub_group_size)

#define GET_FILTER_GS_OIYX_GSV16(prefix, g, o, i, y, x, sub_group_size)  \
    CAT(prefix, _OFFSET) +                                               \
    ((g) % (sub_group_size)) +                                           \
    (sub_group_size)*(                                                   \
        (x)*CAT(prefix, _X_PITCH) +                                      \
        (y)*CAT(prefix, _Y_PITCH) +                                      \
        (i)*CAT(prefix, _IFM_PITCH) +                                    \
        (o)*CAT(prefix, _OFM_PITCH) +                                    \
        ((g) / (sub_group_size))*CAT(prefix, _GROUPS_PITCH)              \
    )

#define GET_FILTER_GS_OIZYX_GSV16(prefix, g, o, i, z, y, x, sub_group_size)  \
    CAT(prefix, _OFFSET) +                                                   \
    ((g) % (sub_group_size)) +                                               \
    (sub_group_size)*(                                                       \
        (x)*CAT(prefix, _X_PITCH) +                                          \
        (y)*CAT(prefix, _Y_PITCH) +                                          \
        (z)*CAT(prefix, _Z_PITCH) +                                          \
        (i)*CAT(prefix, _IFM_PITCH) +                                        \
        (o)*CAT(prefix, _OFM_PITCH) +                                        \
        ((g) / (sub_group_size))*CAT(prefix, _GROUPS_PITCH)                  \
    )

#define GET_FILTER_G_OS_IYX_OSV16_ROTATE_180(prefix, g, o, i, y, x, sub_group_size) \
    CAT(prefix, _OFFSET) +                                                          \
    (g * CAT(prefix, _GROUPS_PITCH)) +                                              \
    ((o) % (sub_group_size)) +                                                      \
    (sub_group_size)*(                                                              \
        (CAT(prefix, _SIZE_X ) - x - 1)*CAT(prefix, _X_PITCH) +                     \
        (CAT(prefix, _SIZE_Y ) - y - 1)*CAT(prefix, _Y_PITCH) +                     \
        (i)*CAT(prefix, _IFM_PITCH) +                                               \
        ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH)                            \
    )

#define GET_FILTER_G_IS_OS_ZYX_ISV16_OSV16_INDEX(prefix, g, o, i, z, y, x, sub_group_size) \
    CAT(prefix, _OFFSET) +                                                                 \
    (g)*CAT(prefix, _GROUPS_PITCH) +                                                       \
    ((o) % (sub_group_size)) +                                                             \
    (sub_group_size)*(                                                                     \
        (x)*(sub_group_size)*CAT(prefix, _X_PITCH) +                                       \
        (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) +                                       \
        (z)*(sub_group_size)*CAT(prefix, _Z_PITCH) +                                       \
        ((i) % (sub_group_size)) +                                                         \
        ((o) / (sub_group_size))*(sub_group_size)*CAT(prefix, _OFM_PITCH) +                \
        ((i) / (sub_group_size))*CAT(prefix, _IFM_PITCH)                                   \
    )

#define GET_FILTER_G_IS_OS_YX_ISV16_OSV16_INDEX(prefix, g, o, i, y, x, sub_group_size) \
    CAT(prefix, _OFFSET) +                                                             \
    (g)*CAT(prefix, _GROUPS_PITCH) +                                                   \
    ((o) % (sub_group_size)) +                                                         \
    (sub_group_size)*(                                                                 \
        (x)*(sub_group_size)*CAT(prefix, _X_PITCH) +                                   \
        (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) +                                   \
        ((i) % (sub_group_size)) +                                                     \
        ((o) / (sub_group_size))*(sub_group_size)*CAT(prefix, _OFM_PITCH) +            \
        ((i) / (sub_group_size))*CAT(prefix, _IFM_PITCH)                               \
    )

#define GET_FILTER_G_OS_IS_ZYX_ISV16_OSV16_INDEX(prefix, g, o, i, z, y, x, sub_group_size) \
    CAT(prefix, _OFFSET) +                                                                 \
    (g)*CAT(prefix, _GROUPS_PITCH) +                                                       \
    ((o) % (sub_group_size)) +                                                             \
    (sub_group_size)*(                                                                     \
        (x)*(sub_group_size)*CAT(prefix, _X_PITCH) +                                       \
        (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) +                                       \
        (z)*(sub_group_size)*CAT(prefix, _Z_PITCH) +                                       \
        ((i) % (sub_group_size)) +                                                         \
        ((i) / (sub_group_size))*(sub_group_size)*CAT(prefix, _IFM_PITCH) +                \
        ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH)                                   \
    )

#define GET_FILTER_GI_YXS_OS_YXSV2_OSV_INDEX(prefix, g, o, i, y, x, sub_group_size) \
    get_gi_yxs_os_yxsv2_osv_index(                                                  \
        g, o, i, y, x,                                                              \
        CAT(prefix, _SIZE_X ),                                                      \
        CAT(prefix, _GROUPS_PITCH),                                                 \
        CAT(prefix, _IFM_PITCH),                                                    \
        CAT(prefix, _Y_PITCH),                                                      \
        CAT(prefix, _X_PITCH),                                                      \
        CAT(prefix, _OFFSET),                                                       \
        sub_group_size)

#define GET_FILTER_GIY_XS_OS_XSV2_OSV_INDEX(prefix, g, o, i, y, x, sub_group_size)  \
    get_giy_xs_os_xsv2_osv_index(                                                   \
        g, o, i, y, x,                                                              \
        CAT(prefix, _SIZE_X ),                                                      \
        CAT(prefix, _GROUPS_PITCH),                                                 \
        CAT(prefix, _IFM_PITCH),                                                    \
        CAT(prefix, _Y_PITCH),                                                      \
        CAT(prefix, _X_PITCH),                                                      \
        CAT(prefix, _OFFSET),                                                       \
        sub_group_size)

inline uint get_gs_oi_yxs_gsv_yxsv4_index(uint g, uint o, uint i, uint y, uint x, uint o_size, uint i_size, uint size_x, uint size_y, const uint gsv) {
    const uint yxsv = 4;
    uint yx = y * size_x + x;
    uint yx_size_aligned = (size_x * size_y + yxsv - 1) / yxsv * yxsv;
    uint gs_index = g / gsv;
    uint yxs_index = yx / yxsv;
    uint gsv_index = g % gsv;
    uint yxsv_index = yx % yxsv;

    uint index = 0;
    index += yxsv_index;
    index += gsv_index * yxsv;
    index += yxs_index * yxsv * gsv;
    index += o * i * gsv * yx_size_aligned;
    index += gs_index * gsv * yx_size_aligned * o_size * i_size;
    return index;
}

#define GET_FILTER_GS_OI_YXS_GSV4_YXSV4_INDEX(prefix, g, o, i, y, x) \
    get_gs_oi_yxs_gsv_yxsv4_index(                                   \
        g, o, i, y, x,                                               \
        CAT(prefix, _OFM_NUM),                                       \
        CAT(prefix, _IFM_NUM),                                       \
        CAT(prefix, _SIZE_X),                                        \
        CAT(prefix, _SIZE_Y),                                        \
        4)

#define GET_FILTER_GS_OI_YXS_GSV16_YXSV4_INDEX(prefix, g, o, i, y, x) \
    get_gs_oi_yxs_gsv_yxsv4_index(                                    \
        g, o, i, y, x,                                                \
        CAT(prefix, _OFM_NUM),                                        \
        CAT(prefix, _IFM_NUM),                                        \
        CAT(prefix, _SIZE_X),                                         \
        CAT(prefix, _SIZE_Y),                                         \
        16)

#define GET_FILTER_GS_OI_YXS_GSV32_YXSV4_INDEX(prefix, g, o, i, y, x) \
    get_gs_oi_yxs_gsv_yxsv4_index(                                    \
        g, o, i, y, x,                                                \
        CAT(prefix, _OFM_NUM),                                        \
        CAT(prefix, _IFM_NUM),                                        \
        CAT(prefix, _SIZE_X),                                         \
        CAT(prefix, _SIZE_Y),                                         \
        32)

#define GET_FILTER_G_OS_IS_YX_ISV16_OSV16_INDEX(prefix, g, o, i, y, x, sub_group_size) \
    CAT(prefix, _OFFSET) +                                                             \
    (g * CAT(prefix, _GROUPS_PITCH)) +                                                 \
    ((o) % (sub_group_size)) +                                                         \
    (sub_group_size)*(                                                                 \
        (x)*(sub_group_size)*CAT(prefix, _X_PITCH) +                                   \
        (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) +                                   \
        ((i) % (sub_group_size)) +                                                     \
        ((i) / (sub_group_size))*(sub_group_size)*CAT(prefix, _IFM_PITCH) +            \
        ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH)                               \
    )

inline uint get_g_os_zyx_is_osv_isv_index(uint g, uint o, uint i, uint z, uint y, uint x,
                                                uint g_size, uint o_size, uint i_size, uint z_size, uint y_size, uint x_size,
                                                uint osv, uint isv) {
    uint is_size = (i_size + isv - 1) / isv;
    uint os_size = (o_size + osv - 1) / osv;

    uint isv_index = i % isv;
    uint osv_index = o % osv;
    uint is_index = i / isv;
    uint os_index = o / osv;

    uint isv_pitch = 1;
    uint osv_pitch = isv_pitch * isv;
    uint is_pitch = osv_pitch * osv;
    uint x_pitch = is_pitch * is_size;
    uint y_pitch = x_pitch * x_size;
    uint z_pitch = y_pitch * y_size;
    uint os_pitch = z_pitch * z_size;
    uint g_pitch = os_pitch * os_size;

    uint index = 0;
    index += isv_index * isv_pitch;
    index += osv_index * osv_pitch;
    index += is_index * is_pitch;
    index += x * x_pitch;
    index += y * y_pitch;
    index += z * z_pitch;
    index += os_index * os_pitch;
    index += g * g_pitch;
    return index;
}

#define GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, osv, isv)            \
    get_g_os_zyx_is_osv_isv_index(                                                          \
    g, o, i, z, y, x,                                                                       \
    CAT(tensor, _GROUPS_NUM),                                                               \
    CAT(tensor, _OFM_NUM),                                                                  \
    CAT(tensor, _IFM_NUM),                                                                  \
    CAT(tensor, _SIZE_Z),                                                                   \
    CAT(tensor, _SIZE_Y),                                                                   \
    CAT(tensor, _SIZE_X),                                                                   \
    osv, isv)

#define GET_FILTER_G_OS_ZYX_IS_OSV16_ISV4_INDEX(tensor, g, o, i, z, y, x)   GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, 16, 4)
#define GET_FILTER_G_OS_ZYX_IS_OSV16_ISV16_INDEX(tensor, g, o, i, z, y, x)  GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, 16, 16)
#define GET_FILTER_G_OS_ZYX_IS_OSV16_ISV32_INDEX(tensor, g, o, i, z, y, x)  GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, 16, 32)
#define GET_FILTER_G_OS_ZYX_IS_OSV32_ISV4_INDEX(tensor, g, o, i, z, y, x)   GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, 32, 4)
#define GET_FILTER_G_OS_ZYX_IS_OSV32_ISV16_INDEX(tensor, g, o, i, z, y, x)  GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, 32, 16)
#define GET_FILTER_G_OS_ZYX_IS_OSV32_ISV32_INDEX(tensor, g, o, i, z, y, x)  GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, 32, 32)

inline uint get_g_os_y_is_x_osv_isv_index(uint g, uint o, uint i, uint y, uint x,
    uint x_size, uint y_size, uint i_size, uint o_size, uint osv_size, uint isv_size)
{
    const uint isv = i % isv_size;
    const uint osv = o % osv_size;
    const uint is = i / isv_size;
    const uint os = o / osv_size;

    const uint x_pitch = osv_size * isv_size;
    const uint is_pitch = x_pitch * x_size;
    const uint y_pitch = is_pitch * ((i_size + isv_size - 1) / isv_size);
    const uint os_pitch = y_pitch * y_size;
    const uint g_pitch = os_pitch * ((o_size + osv_size - 1) / osv_size);

    const uint output_offset =
        isv +
        osv * isv_size +
        x * x_pitch +
        is * is_pitch +
        y * y_pitch +
        os * os_pitch +
        g * g_pitch;

    return output_offset;
}

#define GET_FILTER_G_OS_Y_IS_X_OSV_ISV_INDEX(tensor, g, o, i, y, x, osv, isv)               \
    get_g_os_y_is_x_osv_isv_index(                                                          \
    g, o, i, y, x,                                                                          \
    CAT(tensor, _SIZE_X),                                                                   \
    CAT(tensor, _SIZE_Y),                                                                   \
    CAT(tensor, _IFM_NUM),                                                                  \
    CAT(tensor, _OFM_NUM),                                                                  \
    osv, isv)

inline uint get_g_os_zy_is_x_osv_isv_index(uint g, uint o, uint i, uint z, uint y, uint x,
                                                uint o_size, uint i_size, uint z_size, uint y_size, uint x_size,
                                                uint osv, uint isv) {
    uint is_size = (i_size + isv - 1) / isv;
    uint os_size = (o_size + osv - 1) / osv;

    uint isv_index = i % isv;
    uint osv_index = o % osv;
    uint is_index = i / isv;
    uint os_index = o / osv;

    uint isv_pitch = 1;
    uint osv_pitch = isv_pitch * isv;
    uint x_pitch = osv_pitch * osv;
    uint is_pitch = x_pitch * x_size;
    uint y_pitch = is_pitch * is_size;
    uint z_pitch = y_pitch * y_size;
    uint os_pitch = z_pitch * z_size;
    uint g_pitch = os_pitch * os_size;

    uint index = 0;
    index += isv_index * isv_pitch;
    index += osv_index * osv_pitch;
    index += is_index * is_pitch;
    index += x * x_pitch;
    index += y * y_pitch;
    index += z * z_pitch;
    index += os_index * os_pitch;
    index += g * g_pitch;
    return index;
}

#define GET_FILTER_G_OS_ZY_IS_X_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, osv, isv)        \
    get_g_os_zy_is_x_osv_isv_index(                                                      \
    g, o, i, z, y, x,                                                                    \
    CAT(tensor, _OFM_NUM),                                                               \
    CAT(tensor, _IFM_NUM),                                                               \
    CAT(tensor, _SIZE_Z),                                                                \
    CAT(tensor, _SIZE_Y),                                                                \
    CAT(tensor, _SIZE_X),                                                                \
    osv, isv)

