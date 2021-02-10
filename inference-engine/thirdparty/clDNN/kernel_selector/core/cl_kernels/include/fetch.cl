/*
// Copyright (c) 2016-2020 Intel Corporation
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

#define GET_DATA_INDEX_RAW(prefix, i0, i1, i2, i3)                               \
    CAT(prefix, _OFFSET) +                                             \
    (i0)*CAT(prefix, _PITCHES)[0] + \
    (i1)*CAT(prefix, _PITCHES)[1] + \
    (i2)*CAT(prefix, _PITCHES)[2] + \
    (i3)*CAT(prefix, _PITCHES)[3]

#define GET_DATA_INDEX_SAFE(prefix, b, f, y, x)                     \
    CAT(prefix, _OFFSET) +                                          \
    (x % CAT(prefix, _SIZE_X     ))*CAT(prefix, _X_PITCH) +         \
    (y % CAT(prefix, _SIZE_Y     ))*CAT(prefix, _Y_PITCH) +         \
    (f % CAT(prefix, _FEATURE_NUM))*CAT(prefix, _FEATURE_PITCH) +   \
    (b % CAT(prefix, _BATCH_NUM  ))*CAT(prefix, _BATCH_PITCH)

 #define GET_DATA_INDEX_5D(prefix, b, f, z, y, x) \
    CAT(prefix, _OFFSET) +                  \
    (x)*CAT(prefix, _X_PITCH) +             \
    (y)*CAT(prefix, _Y_PITCH) +             \
    (z)*CAT(prefix, _Z_PITCH) +             \
    (f)*CAT(prefix, _FEATURE_PITCH) +       \
    (b)*CAT(prefix, _BATCH_PITCH)

#define GET_DATA_INDEX_RAW_5D(prefix, i0, i1, i2, i3, i4) \
    CAT(prefix, _OFFSET) + \
    (i0)*CAT(prefix, _PITCHES)[0] + \
    (i1)*CAT(prefix, _PITCHES)[1] + \
    (i2)*CAT(prefix, _PITCHES)[2] + \
    (i3)*CAT(prefix, _PITCHES)[3] + \
    (i4)*CAT(prefix, _PITCHES)[4]

#define GET_DATA_INDEX_5D_SAFE(prefix, b, f, z, y, x)               \
    CAT(prefix, _OFFSET) +                                          \
    (x % CAT(prefix, _SIZE_X     ))*CAT(prefix, _X_PITCH) +         \
    (y % CAT(prefix, _SIZE_Y     ))*CAT(prefix, _Y_PITCH) +         \
    (z % CAT(prefix, _SIZE_Z     ))*CAT(prefix, _Z_PITCH) +         \
    (f % CAT(prefix, _FEATURE_NUM))*CAT(prefix, _FEATURE_PITCH) +   \
    (b % CAT(prefix, _BATCH_NUM  ))*CAT(prefix, _BATCH_PITCH)

#define GET_DATA_INDEX_6D(prefix, b, f, w, z, y, x)     \
    CAT(prefix, _OFFSET) +                              \
    (x)*CAT(prefix, _X_PITCH) +                         \
    (y)*CAT(prefix, _Y_PITCH) +                         \
    (z)*CAT(prefix, _Z_PITCH) +                         \
    (w)*CAT(prefix, _W_PITCH) +                         \
    (f)*CAT(prefix, _FEATURE_PITCH) +                   \
    (b)*CAT(prefix, _BATCH_PITCH)

#define GET_DATA_INDEX_6D_SAFE(prefix, b, f, w, z, y, x)            \
    CAT(prefix, _OFFSET) +                                          \
    (x % CAT(prefix, _SIZE_X     ))*CAT(prefix, _X_PITCH) +         \
    (y % CAT(prefix, _SIZE_Y     ))*CAT(prefix, _Y_PITCH) +         \
    (z % CAT(prefix, _SIZE_Z     ))*CAT(prefix, _Z_PITCH) +         \
    (w % CAT(prefix, _SIZE_W     ))*CAT(prefix, _W_PITCH) +         \
    (f % CAT(prefix, _FEATURE_NUM))*CAT(prefix, _FEATURE_PITCH) +   \
    (b % CAT(prefix, _BATCH_NUM  ))*CAT(prefix, _BATCH_PITCH)

#define GET_DATA_INDEX_RAW_6D(prefix, i0, i1, i2, i3, i4, i5) \
    CAT(prefix, _OFFSET) + \
    (i0)*CAT(prefix, _PITCHES)[0] + \
    (i1)*CAT(prefix, _PITCHES)[1] + \
    (i2)*CAT(prefix, _PITCHES)[2] + \
    (i3)*CAT(prefix, _PITCHES)[3] + \
    (i4)*CAT(prefix, _PITCHES)[4] + \
    (i5)*CAT(prefix, _PITCHES)[5]


#define GET_DATA_BS_FYX_BSV8_INDEX(prefix, b, f, y, x, sub_group_size)  \
    CAT(prefix, _OFFSET) +                                              \
    ((b) % (sub_group_size)) +                                          \
    (sub_group_size)*(                                                  \
        (x)*CAT(prefix, _X_PITCH) +                                     \
        (y)*CAT(prefix, _Y_PITCH) +                                     \
        (f)*CAT(prefix, _FEATURE_PITCH) +                               \
        ((b) / (sub_group_size))*CAT(prefix, _BATCH_PITCH)              \
    )

inline uint FUNC(get_b_fs_yx_fsv_index)(uint b, uint f, uint y, uint x,
                                        uint x_size, uint y_size, uint f_size, uint b_size,
                                        uint b_pad_before, uint b_pad_after,
                                        uint f_pad_before, uint f_pad_after,
                                        uint y_pad_before, uint y_pad_after,
                                        uint x_pad_before, uint x_pad_after, uint alignment) {
    const uint feature = f + f_pad_before;
    const uint fs = feature / alignment;
    const uint fsv = feature % alignment;
    const uint x_pitch = alignment;
    const uint y_pitch = x_pitch * (x_pad_before +  x_size + x_pad_after);
    const uint total_f_size = f_pad_before + f_size + f_pad_after;
    const uint fs_pitch = y_pitch * (y_pad_before +  y_size + y_pad_after);
    const uint b_pitch = fs_pitch * ((total_f_size + alignment - 1) / alignment);

    const uint output_offset =  (b_pad_before + b) * b_pitch +
                                fs * fs_pitch +
                                (y_pad_before + y) * y_pitch +
                                (x_pad_before + x) * x_pitch
                                + fsv;

    return output_offset;
}

inline uint FUNC(get_b_fs_yx_fsv_index_safe)(uint b, uint f, uint y, uint x,
                                             uint x_size, uint y_size, uint f_size, uint b_size,
                                             uint b_pad_before, uint b_pad_after,
                                             uint f_pad_before, uint f_pad_after,
                                             uint y_pad_before, uint y_pad_after,
                                             uint x_pad_before, uint x_pad_after, uint alignment) {
    const uint f_mod = f_pad_before + (f % f_size);
    const uint fs = f_mod / alignment;
    const uint fsv = f_mod % alignment;
    const uint x_pitch = alignment;
    const uint y_pitch = x_pitch * (x_pad_before +  x_size + x_pad_after);
    const uint total_f_size = f_pad_before + f_size + f_pad_after;
    const uint fs_pitch = y_pitch * (y_pad_before +  y_size + y_pad_after);
    const uint b_pitch = fs_pitch * ((total_f_size + alignment - 1) / alignment);

    const uint output_offset = (b_pad_before + (b % b_size)) * b_pitch +
                               fs * fs_pitch +
                               (y_pad_before + (y % y_size)) * y_pitch +
                               (x_pad_before + (x % x_size)) * x_pitch
                               + fsv;

    return output_offset;
}

#define GET_DATA_B_FS_YX_FSV16_INDEX(prefix, b, f, y, x) \
    FUNC_CALL(get_b_fs_yx_fsv_index)(                    \
        b, f, y, x,                                      \
        CAT(prefix, _SIZE_X ),                           \
        CAT(prefix, _SIZE_Y),                            \
        CAT(prefix, _FEATURE_NUM),                       \
        CAT(prefix, _BATCH_NUM),                         \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),              \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),               \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),            \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                 \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                  \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                 \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16)

#define GET_DATA_B_FS_YX_FSV16_INDEX_SAFE(prefix, b, f, y, x) \
    FUNC_CALL(get_b_fs_yx_fsv_index_safe)(                    \
        b, f, y, x,                                           \
        CAT(prefix, _SIZE_X ),                                \
        CAT(prefix, _SIZE_Y),                                 \
        CAT(prefix, _FEATURE_NUM),                            \
        CAT(prefix, _BATCH_NUM),                              \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                   \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                    \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                 \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                  \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                      \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                       \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                      \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16)

#define GET_DATA_B_FS_YX_FSV4_INDEX(prefix, b, f, y, x) \
    FUNC_CALL(get_b_fs_yx_fsv_index)(                   \
        b, f, y, x,                                     \
        CAT(prefix, _SIZE_X ),                          \
        CAT(prefix, _SIZE_Y),                           \
        CAT(prefix, _FEATURE_NUM),                      \
        CAT(prefix, _BATCH_NUM),                        \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),             \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),              \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),           \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),            \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                 \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                \
        CAT(prefix, _PAD_AFTER_SIZE_X), 4)

#define GET_DATA_B_FS_YX_FSV4_INDEX_SAFE(prefix, b, f, y, x) \
    FUNC_CALL(get_b_fs_yx_fsv_index_safe)(                   \
        b, f, y, x,                                          \
        CAT(prefix, _SIZE_X ),                               \
        CAT(prefix, _SIZE_Y),                                \
        CAT(prefix, _FEATURE_NUM),                           \
        CAT(prefix, _BATCH_NUM),                             \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                  \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                   \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                 \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                     \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                     \
        CAT(prefix, _PAD_AFTER_SIZE_X), 4)

#define GET_DATA_B_FS_YX_FSV32_INDEX(prefix, b, f, y, x) \
    FUNC_CALL(get_b_fs_yx_fsv_index)(                    \
        b, f, y, x,                                      \
        CAT(prefix, _SIZE_X ),                           \
        CAT(prefix, _SIZE_Y),                            \
        CAT(prefix, _FEATURE_NUM),                       \
        CAT(prefix, _BATCH_NUM),                         \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),              \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),               \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),            \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                 \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                  \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                 \
        CAT(prefix, _PAD_AFTER_SIZE_X), 32)

#define GET_DATA_B_FS_YX_FSV32_INDEX_SAFE(prefix, b, f, y, x) \
    FUNC_CALL(get_b_fs_yx_fsv_index_safe)(                    \
        b, f, y, x,                                           \
        CAT(prefix, _SIZE_X ),                                \
        CAT(prefix, _SIZE_Y),                                 \
        CAT(prefix, _FEATURE_NUM),                            \
        CAT(prefix, _BATCH_NUM),                              \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                   \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                    \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                 \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                  \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                      \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                       \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                      \
        CAT(prefix, _PAD_AFTER_SIZE_X), 32)

#define GET_FILTER_OS_IS_YX_ISV16_OSV16_INDEX(prefix, o, i, y, x, sub_group_size) \
    CAT(prefix, _OFFSET) +                                                        \
    ((o) % (sub_group_size)) +                                                    \
    (sub_group_size)*(                                                            \
        (x)*(sub_group_size)*CAT(prefix, _X_PITCH) +                              \
        (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) +                              \
        ((i) % (sub_group_size)) +                                                \
        ((i) / (sub_group_size))*(sub_group_size)*CAT(prefix, _IFM_PITCH) +       \
        ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH)                          \
    )

#define GET_FILTER_OS_IS_ZYX_ISV16_OSV16_INDEX(prefix, o, i, z, y, x, sub_group_size) \
    CAT(prefix, _OFFSET) +                                                            \
    ((o) % (sub_group_size)) +                                                        \
    (sub_group_size)*(                                                                \
        (x)*(sub_group_size)*CAT(prefix, _X_PITCH) +                                  \
        (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) +                                  \
        (z)*(sub_group_size)*CAT(prefix, _Z_PITCH) +                                  \
        ((i) % (sub_group_size)) +                                                    \
        ((i) / (sub_group_size))*(sub_group_size)*CAT(prefix, _IFM_PITCH) +           \
        ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH)                              \
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
    FUNC_CALL(get_os_is_zyx_isv8_osv16_isv2_index)(                                   \
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
    FUNC_CALL(get_os_is_zyx_isv8_osv16_isv2_index)(                                       \
        0, o, i, z, y, x,                                                                 \
        CAT(prefix, _SIZE_X),                                                             \
        CAT(prefix, _SIZE_Y),                                                             \
        CAT(prefix, _SIZE_Z),                                                             \
        CAT(prefix, _GROUPS_NUM),                                                         \
        CAT(prefix, _OFM_NUM),                                                            \
        CAT(prefix, _IFM_NUM),                                                            \
        CAT(prefix, _OFFSET)                                                              \
    )

inline uint FUNC(get_os_is_zyx_osv_isv_index)(uint o, uint i, uint z, uint y, uint x,
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

inline uint FUNC(get_g_os_is_zyx_osv_isv_index)(uint g, uint o, uint i, uint z, uint y, uint x,
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
    FUNC_CALL(get_g_os_is_zyx_osv_isv_index)(                                \
        g, o, i, z, y, x,                                                    \
        CAT(prefix, _SIZE_X),                                                \
        CAT(prefix, _SIZE_Y),                                                \
        CAT(prefix, _SIZE_Z),                                                \
        CAT(prefix, _IFM_NUM),                                               \
        CAT(prefix, _OFM_NUM),                                               \
        16,                                                                  \
        16)

#define GET_FILTER_OS_IS_YX_OSV16_ISV16_INDEX(prefix, o, i, y, x) \
    FUNC_CALL(get_os_is_zyx_osv_isv_index)(                       \
        o, i, 0, y, x,                                            \
        CAT(prefix, _SIZE_X),                                     \
        CAT(prefix, _SIZE_Y),                                     \
        1,                                                        \
        CAT(prefix, _IFM_NUM),                                    \
        CAT(prefix, _OFM_NUM),                                    \
        16,                                                       \
        16)

#define GET_FILTER_OS_IS_ZYX_OSV16_ISV16_INDEX(prefix, o, i, z, y, x)   \
    FUNC_CALL(get_os_is_zyx_osv_isv_index)(                             \
        o, i, z, y, x,                                                  \
        CAT(prefix, _SIZE_X),                                           \
        CAT(prefix, _SIZE_Y),                                           \
        CAT(prefix, _SIZE_Z),                                           \
        CAT(prefix, _IFM_NUM),                                          \
        CAT(prefix, _OFM_NUM),                                          \
        16,                                                             \
        16)

#define GET_FILTER_OS_IS_ZYX_OSV32_ISV16_INDEX(prefix, o, i, z, y, x)   \
    FUNC_CALL(get_os_is_zyx_osv_isv_index)(                             \
        o, i, z, y, x,                                                  \
        CAT(prefix, _SIZE_X),                                           \
        CAT(prefix, _SIZE_Y),                                           \
        CAT(prefix, _SIZE_Z),                                           \
        CAT(prefix, _IFM_NUM),                                          \
        CAT(prefix, _OFM_NUM),                                          \
        32,                                                             \
        16)

#define GET_FILTER_OS_IS_ZYX_OSV64_ISV16_INDEX(prefix, o, i, z, y, x)   \
    FUNC_CALL(get_os_is_zyx_osv_isv_index)(                             \
        o, i, z, y, x,                                                  \
        CAT(prefix, _SIZE_X),                                           \
        CAT(prefix, _SIZE_Y),                                           \
        CAT(prefix, _SIZE_Z),                                           \
        CAT(prefix, _IFM_NUM),                                          \
        CAT(prefix, _OFM_NUM),                                          \
        64,                                                             \
        16)

#define GET_FILTER_G_OS_IS_YX_ISV8_OSV16_ISV2_INDEX(prefix, g, o, i, y, x, sub_group_size) \
    FUNC_CALL(get_os_is_zyx_isv8_osv16_isv2_index)(                                        \
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
    FUNC_CALL(get_os_is_zyx_isv8_osv16_isv2_index)(                                            \
        g, o, i, z, y, x,                                                                      \
        CAT(prefix, _SIZE_X),                                                                  \
        CAT(prefix, _SIZE_Y),                                                                  \
        CAT(prefix, _SIZE_Z),                                                                  \
        CAT(prefix, _GROUPS_NUM),                                                              \
        CAT(prefix, _OFM_NUM),                                                                 \
        CAT(prefix, _IFM_NUM),                                                                 \
        CAT(prefix, _OFFSET)                                                                   \
    )

inline uint FUNC(get_os_is_zyx_isv8_osv16_isv2_index)(uint g, uint o, uint i,  uint z, uint y, uint x, uint x_size, uint y_size, uint z_size,
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

inline uint FUNC(get_os_zyxi_osv16_index)(uint o, uint i, uint z, uint y, uint x, uint i_size, uint o_size, uint x_size, uint y_size, uint z_size)
{
    const size_t idx = o%16 + (o / 16)*i_size*x_size*y_size*z_size*16 +
                       16 *(i+ x*i_size + y*i_size*x_size + z*i_size*x_size*y_size);
    return idx;
}

#define GET_FILTER_OS_ZYXI_OSV16(prefix, o, i, z, y, x) \
    FUNC_CALL(get_os_zyxi_osv16_index)(                 \
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

#define GET_FILTER_OS_IYX_OSV8_INDEX(prefix, o, i, y, x, sub_group_size) \
    CAT(prefix, _OFFSET) +                                               \
    ((o) % (sub_group_size)) +                                           \
    (sub_group_size)*(                                                   \
        (x)*CAT(prefix, _X_PITCH) +                                      \
        (y)*CAT(prefix, _Y_PITCH) +                                      \
        (i)*CAT(prefix, _IFM_PITCH) +                                    \
        ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH)                 \
    )

#define GET_FILTER_OS_IYX_OSV8_ROTATE_180_INDEX(prefix, o, i, y, x, sub_group_size) \
    CAT(prefix, _OFFSET) +                                                          \
    ((o) % (sub_group_size)) +                                                      \
    (sub_group_size)*(                                                              \
        (CAT(prefix, _SIZE_X ) - x - 1)*CAT(prefix, _X_PITCH) +                     \
        (CAT(prefix, _SIZE_Y ) - y - 1)*CAT(prefix, _Y_PITCH) +                     \
        (i)*CAT(prefix, _IFM_PITCH) +                                               \
        ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH)                            \
    )

inline uint FUNC(get_gi_yxs_os_yxsv2_osv_index)(uint g, uint o, uint i, uint y, uint x, uint x_size, uint g_pitch, uint i_pitch,
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
    FUNC_CALL(get_gi_yxs_os_yxsv2_osv_index)(                                   \
        0, o, i, y, x,                                                          \
        CAT(prefix, _SIZE_X ),                                                  \
        CAT(prefix, _GROUPS_PITCH),                                             \
        CAT(prefix, _IFM_PITCH),                                                \
        CAT(prefix, _Y_PITCH),                                                  \
        CAT(prefix, _X_PITCH),                                                  \
        CAT(prefix, _OFFSET),                                                   \
        sub_group_size)

inline uint FUNC(get_giy_xs_os_xsv2_osv_index)(uint g, uint o, uint i, uint y, uint x, uint x_size, uint g_pitch,
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
    FUNC_CALL(get_giy_xs_os_xsv2_osv_index)(                                    \
        0, o, i, y, x,                                                          \
        CAT(prefix, _SIZE_X ),                                                  \
        CAT(prefix, _GROUPS_PITCH),                                             \
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

#define GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_INDEX(prefix, o, i, y, x) \
    FUNC_CALL(get_os_is_yx_isa8_osv8_isv4_index)(                    \
        o, i, y, x, CAT(prefix, _SIZE_X ),                           \
        CAT(prefix, _SIZE_Y),                                        \
        CAT(prefix, _IFM_NUM),                                       \
        CAT(prefix, _OFM_NUM),                                       \
        CAT(prefix, _OFFSET))

inline uint FUNC(get_os_is_zyx_isa8_osv8_isv4_index)(uint o, uint i, uint z, uint y, uint x,
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
    FUNC_CALL(get_os_is_zyx_isa8_osv8_isv4_index)(                       \
        o, i, z, y, x,                                                   \
        CAT(prefix, _SIZE_X ),                                           \
        CAT(prefix, _SIZE_Y),                                            \
        CAT(prefix, _SIZE_Z),                                            \
        CAT(prefix, _IFM_NUM),                                           \
        CAT(prefix, _OFM_NUM),                                           \
        CAT(prefix, _OFFSET))

inline uint FUNC(get_os_is_yx_isa8_osv16_isv4_index)(uint o, uint i, uint y, uint x, uint size_x, uint size_y, uint size_ifm, uint size_ofm, uint offset)
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
    FUNC_CALL(get_os_is_yx_isa8_osv16_isv4_index)(                    \
        o, i, y, x, CAT(prefix, _SIZE_X ),                            \
        CAT(prefix, _SIZE_Y),                                         \
        CAT(prefix, _IFM_NUM),                                        \
        CAT(prefix, _OFM_NUM),                                        \
        CAT(prefix, _OFFSET))

inline uint FUNC(get_os_is_zyx_isa8_osv16_isv4_index)(uint o, uint i, uint z, uint y, uint x,
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
    FUNC_CALL(get_os_is_zyx_isa8_osv16_isv4_index)(                       \
        o, i, z, y, x,                                                    \
        CAT(prefix, _SIZE_X ),                                            \
        CAT(prefix, _SIZE_Y),                                             \
        CAT(prefix, _SIZE_Z),                                             \
        CAT(prefix, _IFM_NUM),                                            \
        CAT(prefix, _OFM_NUM),                                            \
        CAT(prefix, _OFFSET))

inline uint FUNC(get_os_is_yx_isa8_osv8_isv4_swizzled_by_4_index)(uint o, uint i, uint y, uint x, uint size_x, uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
    const uint o_swizzled = (o % 4) * 8 + ((o % 32) / 4) + (o / 32) * 32;

    const uint f_32_aligned = ((size_ifm + 31)/32) * 32;
	const uint isv2_idx = i % 4;
	const uint osv_idx = o_swizzled % 8;
	const uint isv1_idx = (i / 4) % 8;
	const uint is_idx = i / 32;
	const uint os_idx = o_swizzled / 8;

	size_t idx = offset + isv2_idx + 4 * (osv_idx + 8 * isv1_idx);
	idx += x * 4 * 8 * 8;
	idx += y * size_x * 4 * 8 * 8;
	idx += is_idx * size_y * size_x * 4 * 8 * 8;
	idx += os_idx * (f_32_aligned/32) * size_y * size_x * 4 * 8 * 8;

    return idx;
}

inline uint FUNC(get_os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4_index)(uint o, uint i, uint y, uint x, uint size_x, uint size_y, uint size_ifm, uint size_ofm, uint offset)
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

inline uint FUNC(get_os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4_index)(uint o, uint i, uint z, uint y, uint x,
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

#define GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(prefix, o, i, y, x) \
	FUNC_CALL(get_os_is_yx_isa8_osv8_isv4_swizzled_by_4_index)(                    \
        o, i, y, x, CAT(prefix, _SIZE_X ),                                         \
        CAT(prefix, _SIZE_Y),                                                      \
        CAT(prefix, _IFM_NUM),                                                     \
        CAT(prefix, _OFM_NUM),                                                     \
        CAT(prefix, _OFFSET))

#define GET_FILTER_OS_IS_YX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(prefix, o, i, y, x) \
    FUNC_CALL(get_os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4_index)(                    \
        o, i, y, x,                                                                     \
        CAT(prefix, _SIZE_X),                                                           \
        CAT(prefix, _SIZE_Y),                                                           \
        CAT(prefix, _IFM_NUM),                                                          \
        CAT(prefix, _OFM_NUM),                                                          \
        CAT(prefix, _OFFSET))

#define GET_FILTER_OS_IS_ZYX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(prefix, o, i, z, y, x) \
    FUNC_CALL(get_os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4_index)(                       \
        o, i, z, y, x,                                                                      \
        CAT(prefix, _SIZE_X),                                                               \
        CAT(prefix, _SIZE_Y),                                                               \
        CAT(prefix, _SIZE_Z),                                                               \
        CAT(prefix, _IFM_NUM),                                                              \
        CAT(prefix, _OFM_NUM),                                                              \
        CAT(prefix, _OFFSET))


inline uint FUNC(get_is_o_yx_isv32_index)(uint o, uint i, uint y, uint x, uint i_size, uint o_size, uint x_size, uint y_size)
{
    const uint i_aligned_to_32 = ((i_size + 31) / 32) * 32;
    const uint i_val = i % 32;
    const uint i_slice = i / 32;
    const size_t idx = i_val + 32* (x + x_size * (y + y_size * (o + o_size * i_slice) ) );
    return idx;
}

#define GET_FILTER_IS_O_YX_ISV32(prefix, o, i, y, x) \
    FUNC_CALL(get_is_o_yx_isv32_index)(              \
        o, i, y, x,                                  \
        CAT(prefix, _IFM_NUM),                       \
        CAT(prefix, _OFM_NUM),                       \
        CAT(prefix, _SIZE_X),                        \
        CAT(prefix, _SIZE_Y))

inline uint FUNC(get_is_o32_yx_isv32_swizzled_by_4_index)(uint o, uint i, uint y, uint x, uint i_size, uint o_size, uint x_size, uint y_size)
{
    const uint o_aligned_to_32 = ((o_size + 31) / 32) * 32;
    const uint o_swizzled = (o % 4) * 8 + ((o % 32) / 4) + (o / 32) * 32;
    const uint i_aligned_to_32 = ((i_size + 31) / 32) * 32;
    const uint i_val = i % 32;
    const uint i_slice = i / 32;
    const size_t idx = i_val + 32* (x + x_size * (y + y_size * (o_swizzled + o_aligned_to_32 * i_slice) ) );
    return idx;
}

#define GET_FILTER_IS_O32_YX_ISV32_SWIZZLED_BY_4(prefix, o, i, y, x) \
    FUNC_CALL(get_is_o32_yx_isv32_swizzled_by_4_index)(              \
        o, i, y, x,                                                  \
        CAT(prefix, _IFM_NUM),                                       \
        CAT(prefix, _OFM_NUM),                                       \
        CAT(prefix, _SIZE_X),                                        \
        CAT(prefix, _SIZE_Y))

inline uint FUNC(get_os_is_y_x8_osv8_isv4_index)(uint o, uint i, uint y, uint x, uint i_size, uint o_size, uint x_size, uint y_size)
{
    const uint i_aligned_to_4 = ((i_size + 3) / 4) * 4;
    const uint o_aligned_to_8 = ((o_size + 7) / 8) * 8;
    const uint x_aligned_to_8 = ((x_size + 7) / 8) * 8;
    const uint i_val = i % 4;
    const uint i_slice = i / 4;
    const uint o_val = o % 8;
    const uint o_slice = o / 8;
    const size_t idx = i_val + 4 * (o_val + 8 * ( x + x_aligned_to_8 * (y + y_size * (i_slice + (i_aligned_to_4/4) * (o_slice)))));
    return idx;
}

#define GET_FILTER_OS_IS_Y_X8_OSV8_ISV4(prefix, o, i, y, x) \
    FUNC_CALL(get_os_is_y_x8_osv8_isv4_index)(              \
        o, i, y, x,                                         \
        CAT(prefix, _IFM_NUM),                              \
        CAT(prefix, _OFM_NUM),                              \
        CAT(prefix, _SIZE_X),                               \
        CAT(prefix, _SIZE_Y))

inline uint FUNC(get_os_is_y_x8_osv8_isv4_swizzled_by_4_index)(uint o, uint i, uint y, uint x, uint i_size, uint o_size, uint x_size, uint y_size)
{
    const uint i_aligned_to_4 = ((i_size + 3) / 4) * 4;
    const uint o_aligned_to_8 = ((o_size + 7) / 8) * 8;
    const uint x_aligned_to_8 = ((x_size + 7) / 8) * 8;
    const uint i_val = i % 4;
    const uint i_slice = i / 4;
    const uint o_swizzled = (o % 4) * 8 + ((o % 32) / 4) + (o / 32) * 32;
    const uint o_val = o_swizzled % 8;
    const uint o_slice = o_swizzled / 8;
    const size_t idx = i_val + 4 * (o_val + 8 * ( x + x_aligned_to_8 * (y + y_size * (i_slice + (i_aligned_to_4/4) * (o_slice)))));
    return idx;
}

#define GET_FILTER_OS_IS_Y_X8_OSV8_ISV4_SWIZZLED_BY_4(prefix, o, i, y, x) \
    FUNC_CALL(get_os_is_y_x8_osv8_isv4_swizzled_by_4_index)(              \
        o, i, y, x,                                                       \
        CAT(prefix, _IFM_NUM),                                            \
        CAT(prefix, _OFM_NUM),                                            \
        CAT(prefix, _SIZE_X),                                             \
        CAT(prefix, _SIZE_Y))


#define GET_FILTER_G_OS_IS_YX_OSV16_ISV4_INDEX(prefix, g, o, i, y, x) \
    FUNC_CALL(get_g_os_is_yx_osv16_isv4)(                         \
        g, o, i, y, x,                                            \
        CAT(prefix, _IFM_PITCH),                                 \
        CAT(prefix, _OFM_PITCH),                                 \
        CAT(prefix, _SIZE_X),                                    \
        CAT(prefix, _SIZE_Y),                                    \
        CAT(prefix, _OFM_NUM),                                   \
        CAT(prefix, _IFM_NUM))

inline uint FUNC(get_g_os_is_yx_osv16_isv4)(uint g, uint o, uint i, uint y, uint x,
                                          uint i_size,
                                          uint o_size,
                                          uint x_size,
                                          uint y_size,
                                          uint o_num,
                                          uint i_num)
{
    const uint otd = 16;
    uint out_depth_tile = o / otd;
    uint od             = o - out_depth_tile * otd;
    uint output_slice_size = (o_num + otd - 1) / otd;

    const uint tile = 4;
    uint id_tile = i / tile;
    uint id      = i - id_tile * tile;
    uint input_slice_size = (i_num + tile - 1) / tile;

    uint idx = g * output_slice_size * input_slice_size * y_size * x_size * otd * tile
                                       + out_depth_tile * (o_size / tile) * otd * tile
                                       + id_tile                 * i_size * otd * tile
                                       + y                       * x_size * otd * tile
                                       + x                                * otd * tile
                                       + od                                     * tile
                                       + id;

    return idx;
}

#define GET_FILTER_OS_IS_YX_OSV16_ISV4_INDEX(prefix, o, i, y, x) \
    FUNC_CALL(get_os_is_zyx_osv_isv4)(                           \
        o, i, 0, y, x,                                           \
        CAT(prefix, _IFM_PITCH),                                 \
        CAT(prefix, _OFM_PITCH),                                 \
        CAT(prefix, _SIZE_X),                                    \
        CAT(prefix, _SIZE_Y),                                    \
        16)

#define GET_FILTER_OS_IS_YX_OSV32_ISV4_INDEX(prefix, o, i, y, x) \
    FUNC_CALL(get_os_is_zyx_osv_isv4)(                           \
        o, i, 0, y, x,                                           \
        CAT(prefix, _IFM_PITCH),                                 \
        CAT(prefix, _OFM_PITCH),                                 \
        CAT(prefix, _SIZE_X),                                    \
        CAT(prefix, _SIZE_Y),                                    \
        32)

#define GET_FILTER_OS_IS_ZYX_OSV32_ISV4_INDEX(prefix, o, i, z, y, x) \
    FUNC_CALL(get_os_is_zyx_osv_isv4)(                               \
        o, i, z, y, x,                                               \
        CAT(prefix, _IFM_PITCH),                                     \
        CAT(prefix, _OFM_PITCH),                                     \
        CAT(prefix, _SIZE_X),                                        \
        CAT(prefix, _SIZE_Y),                                        \
        32)

inline uint FUNC(get_os_is_zyx_osv_isv4)(uint o, uint i, uint z, uint y, uint x,
                                         uint i_size,
                                         uint o_size,
                                         uint x_size,
                                         uint y_size,
                                         uint otd)
{
    uint out_depth_tile = o / otd;
    uint od             = o - out_depth_tile * otd;

    const uint tile = 4;
    uint id_tile = i / tile;
    uint id      = i - id_tile * tile;

    uint idx = out_depth_tile * (o_size / tile) * otd * tile
               + id_tile               * i_size * otd * tile
               + z            * y_size * x_size * otd * tile
               + y                     * x_size * otd * tile
               + x                              * otd * tile
               + od                                   * tile
               + id;

    return idx;
}

#define GET_FILTER_OS_IS_YX_OSV32_ISV4_SWIZZLED_BY_2_INDEX(prefix, o, i, y, x) \
    FUNC_CALL(get_os_is_yx_osv32_isv4_swizzled_by_2)(                          \
        o, i, y, x,                                                            \
        CAT(prefix, _OFM_NUM),                                                 \
        CAT(prefix, _IFM_NUM),                                                 \
        CAT(prefix, _SIZE_Y),                                                  \
        CAT(prefix, _SIZE_X))

inline uint FUNC(get_os_is_yx_osv32_isv4_swizzled_by_2)(uint o, uint i, uint y, uint x,
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

#define GET_DATA_FS_B_YX_FSV32_INDEX(prefix, b, f, y, x) \
    FUNC_CALL(get_fs_b_yx_fsv32_index)(                  \
        b, f, y, x,                                      \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                 \
        CAT(prefix, _SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X),                  \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                 \
        CAT(prefix, _SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                  \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),            \
        CAT(prefix, _BATCH_NUM))

inline uint FUNC(get_fs_b_yx_fsv32_index)(uint b, uint f, uint y, uint x,
                                          uint x_pad_before, uint x_size, uint x_pad_after,
                                          uint y_pad_before, uint y_size, uint y_pad_after,
                                          uint f_pad_before,
                                          uint size_b)
{
    const uint feature_tile_size = 32;                             // size of the feature tile (slice)

    const uint x_total_size = x_pad_before + x_size + x_pad_after; // total size of x before padding
    const uint y_total_size = y_pad_before + y_size + y_pad_after; // total size of y before padding

    const uint real_x = x + x_pad_before;                          // x before padding
    const uint real_y = y + y_pad_before;                          // y before padding
    const uint real_f = f + f_pad_before;                          // f before padding

    const uint x_pitch = feature_tile_size;                        // difference in location between (x+1) and (x)
    const uint y_pitch = x_pitch * x_total_size;                   // difference in location between (y+1) and (y)
    const uint b_pitch = y_pitch * y_total_size;                   // difference in location between (b+1) and (b)
    const uint f_tile_pitch = b_pitch * size_b;                    // difference in location between (fs+1) and (fs)

    const uint feature_tile_number = real_f / feature_tile_size;        // number of tile which feature belongs to
    const uint feature_local_number = real_f % feature_tile_size;       // local number of feature in tile

    size_t index = 0;

    index += feature_tile_number * f_tile_pitch; // locate beginning of feature tile
    index += b * b_pitch;                        // locate beginning of batch
    index += real_y * y_pitch;                   // locate beginning of y with respect to padding
    index += real_x * x_pitch;                   // locate beginning of x with respect to padding
    index += feature_local_number;               // find requested index by adding feature location in tile

    return index;
}

#define GET_DATA_B_FS_ZYX_FSV16_INDEX(prefix, b, f, z, y, x) \
    FUNC_CALL(get_b_fs_zyx_fsv_index)(                       \
        b, f, z, y, x,                                       \
        CAT(prefix, _SIZE_X ),                               \
        CAT(prefix, _SIZE_Y),                                \
        CAT(prefix, _SIZE_Z),                                \
        CAT(prefix, _FEATURE_NUM),                           \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                 \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                     \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                     \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                     \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16)

#define GET_DATA_B_FS_ZYX_FSV16_INDEX_SAFE(prefix, b, f, z, y, x) \
    FUNC_CALL(get_b_fs_zyx_fsv_index_safe)(                       \
        b, f, z, y, x,                                            \
        CAT(prefix, _SIZE_X),                                     \
        CAT(prefix, _SIZE_Y),                                     \
        CAT(prefix, _SIZE_Z),                                     \
        CAT(prefix, _FEATURE_NUM),                                \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                     \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                          \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                           \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                          \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                           \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                          \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16)


#define GET_DATA_B_FS_ZYX_FSV32_INDEX(prefix, b, f, z, y, x) \
    FUNC_CALL(get_b_fs_zyx_fsv_index)(                       \
        b, f, z, y, x,                                       \
        CAT(prefix, _SIZE_X ),                               \
        CAT(prefix, _SIZE_Y),                                \
        CAT(prefix, _SIZE_Z),                                \
        CAT(prefix, _FEATURE_NUM),                           \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                 \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                     \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                     \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                     \
        CAT(prefix, _PAD_AFTER_SIZE_X), 32)

#define GET_DATA_B_FS_ZYX_FSV32_INDEX_SAFE(prefix, b, f, z, y, x) \
    FUNC_CALL(get_b_fs_zyx_fsv_index_safe)(                       \
        b, f, z, y, x,                                            \
        CAT(prefix, _SIZE_X),                                     \
        CAT(prefix, _SIZE_Y),                                     \
        CAT(prefix, _SIZE_Z),                                     \
        CAT(prefix, _FEATURE_NUM),                                \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                     \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                          \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                           \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                          \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                           \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                          \
        CAT(prefix, _PAD_AFTER_SIZE_X), 32)

inline uint FUNC(get_b_fs_zyx_fsv_index)(uint b, uint f,  uint z, uint y, uint x,
                                         uint x_size, uint y_size, uint z_size, uint f_size,
                                         uint f_pad_before, uint f_pad_after,
                                         uint z_pad_before, uint z_pad_after,
                                         uint y_pad_before, uint y_pad_after,
                                         uint x_pad_before, uint x_pad_after,
                                         uint alignment)
{
    const uint feature = f + f_pad_before;
    const uint fs = feature / alignment;
    const uint fsv = feature % alignment;
    const uint x_pitch = alignment;
    const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
    const uint z_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
    const uint fs_pitch = z_pitch * (z_pad_before + z_size + z_pad_after);
    const uint total_f_size = f_pad_before + f_size + f_pad_after;
    const uint b_pitch = fs_pitch * ((total_f_size + alignment - 1) / alignment);

    const uint output_offset = b * b_pitch +
                               fs * fs_pitch +
                               (z_pad_before + z) * z_pitch +
                               (y_pad_before + y) * y_pitch +
                               (x_pad_before + x) * x_pitch
                               + fsv;

    return output_offset;
}

inline uint FUNC(get_b_fs_zyx_fsv_index_safe)(uint b, uint f,  uint z, uint y, uint x,
                                              uint x_size, uint y_size, uint z_size, uint f_size,
                                              uint f_pad_before, uint f_pad_after,
                                              uint z_pad_before, uint z_pad_after,
                                              uint y_pad_before, uint y_pad_after,
                                              uint x_pad_before, uint x_pad_after,
                                              uint alignment) {
    const uint f_mod = f_pad_before + (f % f_size);
    const uint fs = f_mod / alignment;
    const uint fsv = f_mod % alignment;
    const uint x_pitch = alignment;
    const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
    const uint z_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
    const uint fs_pitch = z_pitch * (z_pad_before + z_size + z_pad_after);
    const uint total_f_size = f_pad_before + f_size + f_pad_after;
    const uint b_pitch = fs_pitch * ((total_f_size + alignment - 1) / alignment);

    const uint output_offset = b * b_pitch +
                               fs * fs_pitch +
                               (z_pad_before + (z % z_size)) * z_pitch +
                               (y_pad_before + (y % y_size)) * y_pitch +
                               (x_pad_before + (x % x_size)) * x_pitch
                               + fsv;

    return output_offset;
}

inline uint FUNC(get_bs_fs_zyx_bsv_fsv_index_safe)(uint b, uint f, uint z, uint y, uint x,
                                                  uint x_size, uint y_size, uint z_size, uint f_size, uint b_size,
                                                  uint f_pad_before, uint f_pad_after,
                                                  uint z_pad_before, uint z_pad_after,
                                                  uint y_pad_before, uint y_pad_after,
                                                  uint x_pad_before, uint x_pad_after, uint alignmentF, uint alignmentB) {
    const uint b_mod = b % b_size;
    const uint f_mod = f_pad_before + (f % f_size);
    const uint fs = f_mod / alignmentF;
    const uint fsv = f_mod % alignmentF;
    const uint bs = b_mod / alignmentB;
    const uint bsv = b_mod % alignmentB;
    const uint x_pitch = alignmentF * alignmentB;
    const uint y_pitch = x_pitch * (x_pad_before +  x_size + x_pad_after);
    const uint z_pitch = y_pitch * (y_pad_before +  y_size + y_pad_after);
    const uint total_f_size = f_pad_before + f_size + f_pad_after;
    const uint fs_pitch = z_pitch * (z_pad_before +  z_size + z_pad_after);
    const uint b_pitch = fs_pitch * ((total_f_size + alignmentF - 1) / alignmentF);

    const uint output_offset = (bs * b_pitch) + (bsv * alignmentF) +
                               fs * fs_pitch +
                               (z_pad_before + (z % z_size)) * z_pitch +
                               (y_pad_before + (y % y_size)) * y_pitch +
                               (x_pad_before + (x % x_size)) * x_pitch
                               + fsv;

    return output_offset;
}

inline uint FUNC(get_bs_fs_zyx_bsv16_fsv16_index)(uint b, uint f,  uint z, uint y, uint x,
                                                  uint x_size, uint y_size, uint z_size, uint f_size,
                                                  uint f_pad_before, uint f_pad_after,
                                                  uint z_pad_before, uint z_pad_after,
                                                  uint y_pad_before, uint y_pad_after,
                                                  uint x_pad_before, uint x_pad_after) {
    const uint alignment = 16;
    const uint feature = f + f_pad_before;
    const uint fs = feature / alignment;
    const uint fsv = feature % alignment;
    const uint bs = b / alignment;
    const uint bsv = b % alignment;

    const uint bsv_pitch = alignment;
    const uint x_pitch = bsv_pitch * alignment;
    const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
    const uint z_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
    const uint fs_pitch = z_pitch * (z_pad_before + z_size + z_pad_after);
    const uint total_f_size = f_pad_before + f_size + f_pad_after;
    const uint bs_pitch = fs_pitch * ((total_f_size + alignment - 1) / alignment);

    const uint output_offset = bs * bs_pitch +
                               fs * fs_pitch +
                               (z_pad_before + z) * z_pitch +
                               (y_pad_before + y) * y_pitch +
                               (x_pad_before + x) * x_pitch +
                               bsv * bsv_pitch
                               + fsv;

    return output_offset;
}

#define GET_DATA_BS_FS_YX_BSV16_FSV16_INDEX(prefix, b, f, y, x) \
    FUNC_CALL(get_bs_fs_zyx_bsv16_fsv16_index)(                     \
        b, f, 0, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X))

#define GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX(prefix, b, f, z, y, x) \
    FUNC_CALL(get_bs_fs_zyx_bsv16_fsv16_index)(                     \
        b, f, z, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X))

#define GET_DATA_BS_FS_YX_BSV16_FSV16_INDEX_SAFE(prefix, b, f, y, x) \
    FUNC_CALL(get_bs_fs_zyx_bsv_fsv_index_safe)(                     \
        b, f, 0, y, x,                                               \
        CAT(prefix, _SIZE_X),                                        \
        CAT(prefix, _SIZE_Y),                                        \
        CAT(prefix, _SIZE_Z),                                        \
        CAT(prefix, _FEATURE_NUM),                                   \
        CAT(prefix, _BATCH_NUM),                                     \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                        \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                         \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                             \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 16)

#define GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX_SAFE(prefix, b, f, z, y, x) \
    FUNC_CALL(get_bs_fs_zyx_bsv_fsv_index_safe)(                         \
        b, f, z, y, x,                                                   \
        CAT(prefix, _SIZE_X ),                                           \
        CAT(prefix, _SIZE_Y),                                            \
        CAT(prefix, _SIZE_Z),                                            \
        CAT(prefix, _FEATURE_NUM),                                       \
        CAT(prefix, _BATCH_NUM),                                         \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                            \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                                 \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                                  \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                                 \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                                  \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                                 \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 16)

inline uint FUNC(get_os_is_osv32_isv32_swizzled_by_4_index)(uint o, uint i, uint y, uint x, uint size_x, uint size_y, uint size_ifm, uint size_ofm, uint offset)
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

#define GET_FILTER_OS_IS_OSV32_ISV32_SWIZZLED_BY_4_INDEX(prefix, o, i, y, x)\
    FUNC_CALL(get_os_is_osv32_isv32_swizzled_by_4_index)(\
        o, i, y, x, CAT(prefix, _SIZE_X ),\
        CAT(prefix, _SIZE_Y),\
        CAT(prefix, _IFM_NUM),\
        CAT(prefix, _OFM_NUM),\
        CAT(prefix, _OFFSET))

inline uint FUNC(get_os_i_yxs_osv_yxsv4_index)(uint o, uint i, uint y, uint x, uint i_size, uint size_x, uint size_y, uint osv) {
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

#define GET_FILTER_OS_I_YXS_OSV4_YXSV4_INDEX(prefix, o, i, y, x)    \
    FUNC_CALL(get_os_i_yxs_osv_yxsv4_index)(                        \
        o, i, y, x,                                                 \
        CAT(prefix, _IFM_NUM),                                      \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        4)

#define GET_FILTER_OS_IYX_OSV32__AI32_INDEX(prefix, o, i, y, x, sub_group_size) \
    CAT(prefix, _OFFSET) +                                                      \
    ((o) % (sub_group_size)) +                                                  \
    (sub_group_size)*(                                                          \
        (x)*CAT(prefix, _X_PITCH) +                                             \
        (y)*CAT(prefix, _Y_PITCH) +                                             \
        (i)*CAT(prefix, _IFM_PITCH) +                                           \
        ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH)                        \
    )

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
    FUNC_CALL(get_gi_yxs_os_yxsv2_osv_index)(                                       \
        g, o, i, y, x,                                                              \
        CAT(prefix, _SIZE_X ),                                                      \
        CAT(prefix, _GROUPS_PITCH),                                                 \
        CAT(prefix, _IFM_PITCH),                                                    \
        CAT(prefix, _Y_PITCH),                                                      \
        CAT(prefix, _X_PITCH),                                                      \
        CAT(prefix, _OFFSET),                                                       \
        sub_group_size)

#define GET_FILTER_GIY_XS_OS_XSV2_OSV_INDEX(prefix, g, o, i, y, x, sub_group_size)  \
    FUNC_CALL(get_giy_xs_os_xsv2_osv_index)(                                        \
        g, o, i, y, x,                                                              \
        CAT(prefix, _SIZE_X ),                                                      \
        CAT(prefix, _GROUPS_PITCH),                                                 \
        CAT(prefix, _IFM_PITCH),                                                    \
        CAT(prefix, _Y_PITCH),                                                      \
        CAT(prefix, _X_PITCH),                                                      \
        CAT(prefix, _OFFSET),                                                       \
        sub_group_size)

inline uint FUNC(get_gs_oi_yxs_gsv_yxsv4_index)(uint g, uint o, uint i, uint y, uint x, uint o_size, uint i_size, uint size_x, uint size_y, const uint gsv) {
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
    FUNC_CALL(get_gs_oi_yxs_gsv_yxsv4_index)(                        \
        g, o, i, y, x,                                               \
        CAT(prefix, _OFM_NUM),                                       \
        CAT(prefix, _IFM_NUM),                                       \
        CAT(prefix, _SIZE_X),                                        \
        CAT(prefix, _SIZE_Y),                                        \
        4)

#define GET_FILTER_GS_OI_YXS_GSV16_YXSV4_INDEX(prefix, g, o, i, y, x) \
    FUNC_CALL(get_gs_oi_yxs_gsv_yxsv4_index)(                         \
        g, o, i, y, x,                                                \
        CAT(prefix, _OFM_NUM),                                        \
        CAT(prefix, _IFM_NUM),                                        \
        CAT(prefix, _SIZE_X),                                         \
        CAT(prefix, _SIZE_Y),                                         \
        16)

#define GET_FILTER_GS_OI_YXS_GSV32_YXSV4_INDEX(prefix, g, o, i, y, x) \
    FUNC_CALL(get_gs_oi_yxs_gsv_yxsv4_index)(                         \
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

inline uint FUNC(get_g_os_zyx_is_osv_isv_index)(uint g, uint o, uint i, uint z, uint y, uint x,
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
    FUNC_CALL(get_g_os_zyx_is_osv_isv_index)(                                               \
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

#define DECLARE_SAMPLER const sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST

#if FP16_UNIT_USED
    #define IMAGE_READ(image, coord) read_imageh((image), imageSampler, (coord))
    #define IMAGE_WRITE(image, coord, val) write_imageh((image), (coord), (val))
#else
    #define IMAGE_READ(image, coord) read_imagef((image), imageSampler, (coord))
    #define IMAGE_WRITE(image, coord, val) write_imagef((image), (coord), (val))
#endif
