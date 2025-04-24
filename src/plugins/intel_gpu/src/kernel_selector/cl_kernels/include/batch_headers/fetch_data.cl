// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common.cl"

// Default formats use <prefix>_OFFSET for batching
#define GET_DATA_INDEX(prefix, b, f, y, x)  \
    CAT(prefix, _OFFSET) +                  \
    (x)*CAT(prefix, _X_PITCH) +             \
    (y)*CAT(prefix, _Y_PITCH) +             \
    (f)*CAT(prefix, _FEATURE_PITCH) +       \
    (b)*CAT(prefix, _BATCH_PITCH)

#define GET_DATA_INDEX_RAW(prefix, i0, i1, i2, i3)                     \
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

#define GET_DATA_INDEX_5D_RAW(prefix, i0, i1, i2, i3, i4) \
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

#define GET_DATA_INDEX_6D_RAW(prefix, i0, i1, i2, i3, i4, i5) \
    CAT(prefix, _OFFSET) + \
    (i0)*CAT(prefix, _PITCHES)[0] + \
    (i1)*CAT(prefix, _PITCHES)[1] + \
    (i2)*CAT(prefix, _PITCHES)[2] + \
    (i3)*CAT(prefix, _PITCHES)[3] + \
    (i4)*CAT(prefix, _PITCHES)[4] + \
    (i5)*CAT(prefix, _PITCHES)[5]

#define GET_DATA_INDEX_7D(prefix, b, f, u, w, z, y, x)  \
    CAT(prefix, _OFFSET) +                              \
    (x)*CAT(prefix, _X_PITCH) +                         \
    (y)*CAT(prefix, _Y_PITCH) +                         \
    (z)*CAT(prefix, _Z_PITCH) +                         \
    (w)*CAT(prefix, _W_PITCH) +                         \
    (u)*CAT(prefix, _U_PITCH) +                         \
    (f)*CAT(prefix, _FEATURE_PITCH) +                   \
    (b)*CAT(prefix, _BATCH_PITCH)

#define GET_DATA_INDEX_7D_SAFE(prefix, b, f, u, w, z, y, x)         \
    CAT(prefix, _OFFSET) +                                          \
    (x % CAT(prefix, _SIZE_X     ))*CAT(prefix, _X_PITCH) +         \
    (y % CAT(prefix, _SIZE_Y     ))*CAT(prefix, _Y_PITCH) +         \
    (z % CAT(prefix, _SIZE_Z     ))*CAT(prefix, _Z_PITCH) +         \
    (w % CAT(prefix, _SIZE_W     ))*CAT(prefix, _W_PITCH) +         \
    (u % CAT(prefix, _SIZE_U     ))*CAT(prefix, _U_PITCH) +         \
    (f % CAT(prefix, _FEATURE_NUM))*CAT(prefix, _FEATURE_PITCH) +   \
    (b % CAT(prefix, _BATCH_NUM  ))*CAT(prefix, _BATCH_PITCH)

#define GET_DATA_INDEX_7D_RAW(prefix, i0, i1, i2, i3, i4, i5, i6) \
    CAT(prefix, _OFFSET) + \
    (i0)*CAT(prefix, _PITCHES)[0] + \
    (i1)*CAT(prefix, _PITCHES)[1] + \
    (i2)*CAT(prefix, _PITCHES)[2] + \
    (i3)*CAT(prefix, _PITCHES)[3] + \
    (i4)*CAT(prefix, _PITCHES)[4] + \
    (i5)*CAT(prefix, _PITCHES)[5] + \
    (i6)*CAT(prefix, _PITCHES)[6]

#define GET_DATA_INDEX_8D(prefix, b, f, v, u, w, z, y, x)   \
    CAT(prefix, _OFFSET) +                                  \
    (x)*CAT(prefix, _X_PITCH) +                             \
    (y)*CAT(prefix, _Y_PITCH) +                             \
    (z)*CAT(prefix, _Z_PITCH) +                             \
    (w)*CAT(prefix, _W_PITCH) +                             \
    (u)*CAT(prefix, _U_PITCH) +                             \
    (v)*CAT(prefix, _V_PITCH) +                             \
    (f)*CAT(prefix, _FEATURE_PITCH) +                       \
    (b)*CAT(prefix, _BATCH_PITCH)

#define GET_DATA_INDEX_8D_SAFE(prefix, b, f, v, u, w, z, y, x)      \
    CAT(prefix, _OFFSET) +                                          \
    (x % CAT(prefix, _SIZE_X     ))*CAT(prefix, _X_PITCH) +         \
    (y % CAT(prefix, _SIZE_Y     ))*CAT(prefix, _Y_PITCH) +         \
    (z % CAT(prefix, _SIZE_Z     ))*CAT(prefix, _Z_PITCH) +         \
    (w % CAT(prefix, _SIZE_W     ))*CAT(prefix, _W_PITCH) +         \
    (u % CAT(prefix, _SIZE_U     ))*CAT(prefix, _U_PITCH) +         \
    (v % CAT(prefix, _SIZE_V     ))*CAT(prefix, _V_PITCH) +         \
    (f % CAT(prefix, _FEATURE_NUM))*CAT(prefix, _FEATURE_PITCH) +   \
    (b % CAT(prefix, _BATCH_NUM  ))*CAT(prefix, _BATCH_PITCH)

#define GET_DATA_INDEX_8D_RAW(prefix, i0, i1, i2, i3, i4, i5, i6, i7) \
    CAT(prefix, _OFFSET) + \
    (i0)*CAT(prefix, _PITCHES)[0] + \
    (i1)*CAT(prefix, _PITCHES)[1] + \
    (i2)*CAT(prefix, _PITCHES)[2] + \
    (i3)*CAT(prefix, _PITCHES)[3] + \
    (i4)*CAT(prefix, _PITCHES)[4] + \
    (i5)*CAT(prefix, _PITCHES)[5] + \
    (i6)*CAT(prefix, _PITCHES)[6] + \
    (i7)*CAT(prefix, _PITCHES)[7]

#define GET_DATA_BS_FYX_BSV8_INDEX(prefix, b, f, y, x, sub_group_size)  \
    CAT(prefix, _OFFSET) +                                              \
    ((b) % (sub_group_size)) +                                          \
    (sub_group_size)*(                                                  \
        (x)*CAT(prefix, _X_PITCH) +                                     \
        (y)*CAT(prefix, _Y_PITCH) +                                     \
        (f)*CAT(prefix, _FEATURE_PITCH) +                               \
        ((b) / (sub_group_size))*CAT(prefix, _BATCH_PITCH)              \
    )

// Blocked formats use pad_before definition for batching
inline uint get_b_fs_yx_fsv_index(uint b, uint f, uint y, uint x,
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

inline uint get_b_fs_yx_fsv_index_safe(uint b, uint f, uint y, uint x,
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
    get_b_fs_yx_fsv_index(                    \
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
    get_b_fs_yx_fsv_index_safe(                    \
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

#define GET_DATA_B_FS_YX_FSV2_INDEX(prefix, b, f, y, x) \
    get_b_fs_yx_fsv_index(                   \
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
        CAT(prefix, _PAD_AFTER_SIZE_X), 2)

#define GET_DATA_B_FS_YX_FSV2_INDEX_SAFE(prefix, b, f, y, x) \
    get_b_fs_yx_fsv_index_safe(                   \
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
        CAT(prefix, _PAD_AFTER_SIZE_X), 2)

#define GET_DATA_B_FS_YX_FSV4_INDEX(prefix, b, f, y, x) \
    get_b_fs_yx_fsv_index(                   \
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
    get_b_fs_yx_fsv_index_safe(                   \
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

#define GET_DATA_B_FS_YX_FSV8_INDEX(prefix, b, f, y, x) \
    get_b_fs_yx_fsv_index(                   \
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
        CAT(prefix, _PAD_AFTER_SIZE_X), 8)

#define GET_DATA_B_FS_YX_FSV8_INDEX_SAFE(prefix, b, f, y, x) \
    get_b_fs_yx_fsv_index_safe(                   \
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
        CAT(prefix, _PAD_AFTER_SIZE_X), 8)

#define GET_DATA_B_FS_YX_FSV32_INDEX(prefix, b, f, y, x) \
    get_b_fs_yx_fsv_index(                    \
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
    get_b_fs_yx_fsv_index_safe(                    \
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


// TODO: No consideration for batch axis padding for batching
inline uint get_fs_b_yx_fsv32_index(uint b, uint f, uint y, uint x,
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

inline uint get_fs_b_yx_fsv32_index_safe(uint b, uint f, uint y, uint x,
                                         uint x_pad_before, uint x_size, uint x_pad_after,
                                         uint y_pad_before, uint y_size, uint y_pad_after,
                                         uint f_pad_before, uint f_size,
                                         uint size_b)
{
    const uint feature_tile_size = 32;                             // size of the feature tile (slice)

    const uint x_total_size = x_pad_before + x_size + x_pad_after; // total size of x before padding
    const uint y_total_size = y_pad_before + y_size + y_pad_after; // total size of y before padding

    const uint real_x = (x % x_size) + x_pad_before;               // x before padding
    const uint real_y = (y % y_size) + y_pad_before;               // y before padding
    const uint real_f = (f % f_size) + f_pad_before;               // f before padding

    const uint x_pitch = feature_tile_size;                        // difference in location between (x+1) and (x)
    const uint y_pitch = x_pitch * x_total_size;                   // difference in location between (y+1) and (y)
    const uint b_pitch = y_pitch * y_total_size;                   // difference in location between (b+1) and (b)
    const uint f_tile_pitch = b_pitch * size_b;                    // difference in location between (fs+1) and (fs)

    const uint feature_tile_number = real_f / feature_tile_size;   // number of tile which feature belongs to
    const uint feature_local_number = real_f % feature_tile_size;  // local number of feature in tile

    size_t index = 0;

    index += feature_tile_number * f_tile_pitch; // locate beginning of feature tile
    index += b * b_pitch;                        // locate beginning of batch
    index += real_y * y_pitch;                   // locate beginning of y with respect to padding
    index += real_x * x_pitch;                   // locate beginning of x with respect to padding
    index += feature_local_number;               // find requested index by adding feature location in tile

    return index;
}

#define GET_DATA_FS_B_YX_FSV32_INDEX(prefix, b, f, y, x) \
    get_fs_b_yx_fsv32_index(                  \
        b, f, y, x,                                      \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                 \
        CAT(prefix, _SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X),                  \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                 \
        CAT(prefix, _SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                  \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),            \
        CAT(prefix, _BATCH_NUM))

#define GET_DATA_FS_B_YX_FSV32_INDEX_SAFE(prefix, b, f, y, x) \
    get_fs_b_yx_fsv32_index_safe(                             \
        b, f, y, x,                                           \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                      \
        CAT(prefix, _SIZE_X),                                 \
        CAT(prefix, _PAD_AFTER_SIZE_X),                       \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                      \
        CAT(prefix, _SIZE_Y),                                 \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                       \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                 \
        CAT(prefix, _FEATURE_NUM),                            \
        CAT(prefix, _BATCH_NUM))


// Blocked formats 5dims use pad_before definition for batching
#define GET_DATA_B_FS_ZYX_FSV2_INDEX(prefix, b, f, z, y, x) \
    get_b_fs_zyx_fsv_index(                                  \
        b, f, z, y, x,                                       \
        CAT(prefix, _SIZE_X ),                               \
        CAT(prefix, _SIZE_Y),                                \
        CAT(prefix, _SIZE_Z),                                \
        CAT(prefix, _FEATURE_NUM),                           \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                  \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                   \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                 \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                     \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                     \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                     \
        CAT(prefix, _PAD_AFTER_SIZE_X), 2)

#define GET_DATA_B_FS_ZYX_FSV2_INDEX_SAFE(prefix, b, f, z, y, x) \
    get_b_fs_zyx_fsv_index_safe(                                  \
        b, f, z, y, x,                                            \
        CAT(prefix, _SIZE_X),                                     \
        CAT(prefix, _SIZE_Y),                                     \
        CAT(prefix, _SIZE_Z),                                     \
        CAT(prefix, _FEATURE_NUM),                                \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                       \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                        \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                     \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                          \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                           \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                          \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                           \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                          \
        CAT(prefix, _PAD_AFTER_SIZE_X), 2)

#define GET_DATA_B_FS_ZYX_FSV4_INDEX(prefix, b, f, z, y, x) \
    get_b_fs_zyx_fsv_index(                                  \
        b, f, z, y, x,                                       \
        CAT(prefix, _SIZE_X ),                               \
        CAT(prefix, _SIZE_Y),                                \
        CAT(prefix, _SIZE_Z),                                \
        CAT(prefix, _FEATURE_NUM),                           \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                  \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                   \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                 \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                     \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                     \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                     \
        CAT(prefix, _PAD_AFTER_SIZE_X), 4)

#define GET_DATA_B_FS_ZYX_FSV4_INDEX_SAFE(prefix, b, f, z, y, x) \
    get_b_fs_zyx_fsv_index_safe(                                  \
        b, f, z, y, x,                                            \
        CAT(prefix, _SIZE_X),                                     \
        CAT(prefix, _SIZE_Y),                                     \
        CAT(prefix, _SIZE_Z),                                     \
        CAT(prefix, _FEATURE_NUM),                                \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                       \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                        \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                     \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                          \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                           \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                          \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                           \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                          \
        CAT(prefix, _PAD_AFTER_SIZE_X), 4)

#define GET_DATA_B_FS_ZYX_FSV8_INDEX(prefix, b, f, z, y, x) \
    get_b_fs_zyx_fsv_index(                                  \
        b, f, z, y, x,                                       \
        CAT(prefix, _SIZE_X ),                               \
        CAT(prefix, _SIZE_Y),                                \
        CAT(prefix, _SIZE_Z),                                \
        CAT(prefix, _FEATURE_NUM),                           \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                  \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                   \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                 \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                     \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                     \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                     \
        CAT(prefix, _PAD_AFTER_SIZE_X), 8)

#define GET_DATA_B_FS_ZYX_FSV8_INDEX_SAFE(prefix, b, f, z, y, x) \
    get_b_fs_zyx_fsv_index_safe(                                  \
        b, f, z, y, x,                                            \
        CAT(prefix, _SIZE_X),                                     \
        CAT(prefix, _SIZE_Y),                                     \
        CAT(prefix, _SIZE_Z),                                     \
        CAT(prefix, _FEATURE_NUM),                                \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                       \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                        \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                     \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                          \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                           \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                          \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                           \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                          \
        CAT(prefix, _PAD_AFTER_SIZE_X), 8)

#define GET_DATA_B_FS_ZYX_FSV16_INDEX(prefix, b, f, z, y, x) \
    get_b_fs_zyx_fsv_index(                                  \
        b, f, z, y, x,                                       \
        CAT(prefix, _SIZE_X ),                               \
        CAT(prefix, _SIZE_Y),                                \
        CAT(prefix, _SIZE_Z),                                \
        CAT(prefix, _FEATURE_NUM),                           \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                  \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                   \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                 \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                     \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                     \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                     \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16)

#define GET_DATA_B_FS_ZYX_FSV16_INDEX_SAFE(prefix, b, f, z, y, x) \
    get_b_fs_zyx_fsv_index_safe(                                  \
        b, f, z, y, x,                                            \
        CAT(prefix, _SIZE_X),                                     \
        CAT(prefix, _SIZE_Y),                                     \
        CAT(prefix, _SIZE_Z),                                     \
        CAT(prefix, _FEATURE_NUM),                                \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                       \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                        \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                     \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                          \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                           \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                          \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                           \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                          \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16)


#define GET_DATA_B_FS_ZYX_FSV32_INDEX(prefix, b, f, z, y, x) \
    get_b_fs_zyx_fsv_index(                                  \
        b, f, z, y, x,                                       \
        CAT(prefix, _SIZE_X ),                               \
        CAT(prefix, _SIZE_Y),                                \
        CAT(prefix, _SIZE_Z),                                \
        CAT(prefix, _FEATURE_NUM),                           \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                  \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                   \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                 \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                     \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                     \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                     \
        CAT(prefix, _PAD_AFTER_SIZE_X), 32)

#define GET_DATA_B_FS_ZYX_FSV32_INDEX_SAFE(prefix, b, f, z, y, x) \
    get_b_fs_zyx_fsv_index_safe(                                  \
        b, f, z, y, x,                                            \
        CAT(prefix, _SIZE_X),                                     \
        CAT(prefix, _SIZE_Y),                                     \
        CAT(prefix, _SIZE_Z),                                     \
        CAT(prefix, _FEATURE_NUM),                                \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                       \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                        \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                     \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                      \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                          \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                           \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                          \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                           \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                          \
        CAT(prefix, _PAD_AFTER_SIZE_X), 32)

inline uint get_b_fs_zyx_fsv_index(uint b, uint f,  uint z, uint y, uint x,
                                         uint x_size, uint y_size, uint z_size, uint f_size,
                                         uint b_pad_before, uint b_pad_after,
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

    const uint output_offset = (b_pad_before + b) * b_pitch +
                               fs * fs_pitch +
                               (z_pad_before + z) * z_pitch +
                               (y_pad_before + y) * y_pitch +
                               (x_pad_before + x) * x_pitch
                               + fsv;

    return output_offset;
}

inline uint get_b_fs_zyx_fsv_index_safe(uint b, uint f,  uint z, uint y, uint x,
                                              uint x_size, uint y_size, uint z_size, uint f_size,
                                              uint b_pad_before, uint b_pad_after,
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

    const uint output_offset = (b_pad_before + b) * b_pitch +
                               fs * fs_pitch +
                               (z_pad_before + (z % z_size)) * z_pitch +
                               (y_pad_before + (y % y_size)) * y_pitch +
                               (x_pad_before + (x % x_size)) * x_pitch
                               + fsv;

    return output_offset;
}

// Double blocked formats use pad_before definition for batching
inline uint get_bs_fs_zyx_bsv_fsv_index_safe(uint b, uint f, uint z, uint y, uint x,
                                                  uint x_size, uint y_size, uint z_size, uint f_size, uint b_size,
                                                  uint b_pad_before, uint b_pad_after,
                                                  uint f_pad_before, uint f_pad_after,
                                                  uint z_pad_before, uint z_pad_after,
                                                  uint y_pad_before, uint y_pad_after,
                                                  uint x_pad_before, uint x_pad_after, uint alignmentB, uint alignmentF) {
    const uint b_mod = b_pad_before + (b % b_size);
    const uint f_mod = f_pad_before + (f % f_size);
    const uint bs = b_mod / alignmentB;
    const uint bsv = b_mod % alignmentB;
    const uint fs = f_mod / alignmentF;
    const uint fsv = f_mod % alignmentF;
    const uint x_pitch = alignmentF * alignmentB;
    const uint y_pitch = x_pitch * (x_pad_before +  x_size + x_pad_after);
    const uint z_pitch = y_pitch * (y_pad_before +  y_size + y_pad_after);
    const uint total_f_size = f_pad_before + f_size + f_pad_after;
    const uint fs_pitch = z_pitch * (z_pad_before +  z_size + z_pad_after);
    const uint bs_pitch = fs_pitch * ((total_f_size + alignmentF - 1) / alignmentF);

    const uint output_offset = bs * bs_pitch +
                               fs * fs_pitch +
                               (z_pad_before + (z % z_size)) * z_pitch +
                               (y_pad_before + (y % y_size)) * y_pitch +
                               (x_pad_before + (x % x_size)) * x_pitch +
                               (bsv * alignmentF)
                               + fsv;

    return output_offset;
}

inline uint get_bs_fs_zyx_bsv_fsv_index(uint b, uint f,  uint z, uint y, uint x,
                                              uint x_size, uint y_size, uint z_size, uint f_size,
                                              uint b_pad_before, uint b_pad_after,
                                              uint f_pad_before, uint f_pad_after,
                                              uint z_pad_before, uint z_pad_after,
                                              uint y_pad_before, uint y_pad_after,
                                              uint x_pad_before, uint x_pad_after,
                                              uint b_alignment, uint f_alignment) {
    const uint feature = f + f_pad_before;
    const uint fs = feature / f_alignment;
    const uint fsv = feature % f_alignment;
    const uint bs = (b + b_pad_before) / b_alignment;
    const uint bsv = (b + b_pad_before) % b_alignment;
    const uint bsv_pitch = f_alignment;
    const uint x_pitch = bsv_pitch * b_alignment;
    const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
    const uint z_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
    const uint fs_pitch = z_pitch * (z_pad_before + z_size + z_pad_after);
    const uint total_f_size = f_pad_before + f_size + f_pad_after;
    const uint bs_pitch = fs_pitch * ((total_f_size + f_alignment - 1) / f_alignment);

    const uint output_offset = bs * bs_pitch +
                               fs * fs_pitch +
                               (z_pad_before + z) * z_pitch +
                               (y_pad_before + y) * y_pitch +
                               (x_pad_before + x) * x_pitch +
                               bsv * bsv_pitch
                               + fsv;

    return output_offset;
}


#define GET_DATA_BS_FS_YX_BSV16_FSV16_INDEX(prefix, b, f, y, x)     \
    get_bs_fs_zyx_bsv_fsv_index(                                    \
        b, f, 0, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                         \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                          \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 16)

#define GET_DATA_BS_FS_YX_BSV16_FSV32_INDEX(prefix, b, f, y, x)     \
    get_bs_fs_zyx_bsv_fsv_index(                                    \
        b, f, 0, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                         \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                          \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 32)
#define GET_DATA_BS_FS_ZYX_BSV32_FSV32_INDEX(prefix, b, f, z, y, x) \
    get_bs_fs_zyx_bsv_fsv_index(                                    \
        b, f, z, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                         \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                          \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X), 32, 32)

#define GET_DATA_BS_FS_YX_BSV32_FSV32_INDEX(prefix, b, f, y, x)     \
    get_bs_fs_zyx_bsv_fsv_index(                                    \
        b, f, 0, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                         \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                          \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X), 32, 32)

#define GET_DATA_BS_FS_YX_BSV4_FSV4_INDEX(prefix, b, f, y, x)       \
    get_bs_fs_zyx_bsv_fsv_index(                                    \
        b, f, 0, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                         \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                          \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X), 4, 4)

#define GET_DATA_BS_FS_ZYX_BSV16_FSV4_INDEX(prefix, b, f, z, y, x)   \
    get_bs_fs_zyx_bsv_fsv_index(                                    \
        b, f, z, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                         \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                          \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 4)

#define GET_DATA_BS_FS_YX_BSV16_FSV4_INDEX(prefix, b, f, y, x)       \
    get_bs_fs_zyx_bsv_fsv_index(                                    \
        b, f, 0, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                         \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                          \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 4)

#define GET_DATA_BS_FS_ZYX_BSV16_FSV8_INDEX(prefix, b, f, z, y, x)   \
    get_bs_fs_zyx_bsv_fsv_index(                                    \
        b, f, z, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                         \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                          \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 8)

#define GET_DATA_BS_FS_YX_BSV16_FSV8_INDEX(prefix, b, f, y, x)       \
    get_bs_fs_zyx_bsv_fsv_index(                                    \
        b, f, 0, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                         \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                          \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 8)

#define GET_DATA_BS_FS_ZYX_BSV8_FSV4_INDEX(prefix, b, f, z, y, x)   \
    get_bs_fs_zyx_bsv_fsv_index(                                    \
        b, f, z, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                         \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                          \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X), 8, 4)

#define GET_DATA_BS_FS_YX_BSV8_FSV4_INDEX(prefix, b, f, y, x)       \
    get_bs_fs_zyx_bsv_fsv_index(                                    \
        b, f, 0, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                         \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                          \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X), 8, 4)

#define GET_DATA_BS_FS_ZYX_BSV16_FSV2_INDEX(prefix, b, f, z, y, x)   \
    get_bs_fs_zyx_bsv_fsv_index(                                    \
        b, f, z, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                         \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                          \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 2)

#define GET_DATA_BS_FS_YX_BSV16_FSV2_INDEX(prefix, b, f, y, x)       \
    get_bs_fs_zyx_bsv_fsv_index(                                    \
        b, f, 0, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                         \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                          \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 2)

#define GET_DATA_BS_FS_ZYX_BSV8_FSV2_INDEX(prefix, b, f, z, y, x)   \
    get_bs_fs_zyx_bsv_fsv_index(                                    \
        b, f, z, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                         \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                          \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X), 8, 2)

#define GET_DATA_BS_FS_YX_BSV8_FSV2_INDEX(prefix, b, f, y, x)       \
    get_bs_fs_zyx_bsv_fsv_index(                                    \
        b, f, 0, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                         \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                          \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X), 8, 2)

#define GET_DATA_BS_FS_YX_BSV4_FSV2_INDEX(prefix, b, f, y, x)       \
    get_bs_fs_zyx_bsv_fsv_index(                                    \
        b, f, 0, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                         \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                          \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X), 4, 2)

#define GET_DATA_BS_FS_ZYX_BSV32_FSV16_INDEX(prefix, b, f, z, y, x) \
    get_bs_fs_zyx_bsv_fsv_index(                                    \
        b, f, z, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                         \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                          \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X), 32, 16)

#define GET_DATA_BS_FS_YX_BSV32_FSV16_INDEX(prefix, b, f, y, x)     \
    get_bs_fs_zyx_bsv_fsv_index(                                    \
        b, f, 0, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                         \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                          \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X), 32, 16)

#define GET_DATA_BS_FS_ZYX_BSV16_FSV32_INDEX(prefix, b, f, z, y, x) \
    get_bs_fs_zyx_bsv_fsv_index(                                    \
        b, f, z, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                         \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                          \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 32)

#define GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX(prefix, b, f, z, y, x) \
    get_bs_fs_zyx_bsv_fsv_index(                                    \
        b, f, z, y, x,                                              \
        CAT(prefix, _SIZE_X),                                       \
        CAT(prefix, _SIZE_Y),                                       \
        CAT(prefix, _SIZE_Z),                                       \
        CAT(prefix, _FEATURE_NUM),                                  \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                         \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                          \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                       \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                        \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                            \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                            \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 16)

#define GET_DATA_BS_FS_YX_BSV16_FSV16_INDEX_SAFE(prefix, b, f, y, x) \
    get_bs_fs_zyx_bsv_fsv_index_safe(                                \
        b, f, 0, y, x,                                               \
        CAT(prefix, _SIZE_X),                                        \
        CAT(prefix, _SIZE_Y),                                        \
        CAT(prefix, _SIZE_Z),                                        \
        CAT(prefix, _FEATURE_NUM),                                   \
        CAT(prefix, _BATCH_NUM),                                     \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                          \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                           \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                        \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                         \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                             \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 16)

#define GET_DATA_BS_FS_ZYX_BSV32_FSV32_INDEX_SAFE(prefix, b, f, z, y, x) \
    get_bs_fs_zyx_bsv_fsv_index_safe(                                    \
        b, f, z, y, x,                                                   \
        CAT(prefix, _SIZE_X),                                            \
        CAT(prefix, _SIZE_Y),                                            \
        CAT(prefix, _SIZE_Z),                                            \
        CAT(prefix, _FEATURE_NUM),                                       \
        CAT(prefix, _BATCH_NUM),                                         \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                              \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                               \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                            \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                                 \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                                  \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                                 \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                                  \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                                 \
        CAT(prefix, _PAD_AFTER_SIZE_X), 32, 32)

#define GET_DATA_BS_FS_YX_BSV32_FSV32_INDEX_SAFE(prefix, b, f, y, x) \
    get_bs_fs_zyx_bsv_fsv_index_safe(                                \
        b, f, 0, y, x,                                               \
        CAT(prefix, _SIZE_X),                                        \
        CAT(prefix, _SIZE_Y),                                        \
        CAT(prefix, _SIZE_Z),                                        \
        CAT(prefix, _FEATURE_NUM),                                   \
        CAT(prefix, _BATCH_NUM),                                     \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                          \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                           \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                        \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                         \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                             \
        CAT(prefix, _PAD_AFTER_SIZE_X), 32, 32)

#define GET_DATA_BS_FS_YX_BSV4_FSV4_INDEX_SAFE(prefix, b, f, y, x)   \
    get_bs_fs_zyx_bsv_fsv_index_safe(                                \
        b, f, 0, y, x,                                               \
        CAT(prefix, _SIZE_X),                                        \
        CAT(prefix, _SIZE_Y),                                        \
        CAT(prefix, _SIZE_Z),                                        \
        CAT(prefix, _FEATURE_NUM),                                   \
        CAT(prefix, _BATCH_NUM),                                     \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                          \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                           \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                        \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                         \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                             \
        CAT(prefix, _PAD_AFTER_SIZE_X), 4, 4)

#define GET_DATA_BS_FS_YX_BSV16_FSV4_INDEX_SAFE(prefix, b, f, y, x)   \
    get_bs_fs_zyx_bsv_fsv_index_safe(                                \
        b, f, 0, y, x,                                               \
        CAT(prefix, _SIZE_X),                                        \
        CAT(prefix, _SIZE_Y),                                        \
        CAT(prefix, _SIZE_Z),                                        \
        CAT(prefix, _FEATURE_NUM),                                   \
        CAT(prefix, _BATCH_NUM),                                     \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                          \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                           \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                        \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                         \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                             \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 4)

#define GET_DATA_BS_FS_ZYX_BSV16_FSV4_INDEX_SAFE(prefix, b, f, z, y, x) \
    get_bs_fs_zyx_bsv_fsv_index_safe(                                  \
        b, f, z, y, x,                                                 \
        CAT(prefix, _SIZE_X),                                          \
        CAT(prefix, _SIZE_Y),                                          \
        CAT(prefix, _SIZE_Z),                                          \
        CAT(prefix, _FEATURE_NUM),                                     \
        CAT(prefix, _BATCH_NUM),                                       \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                            \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                             \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                          \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                           \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                               \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                                \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                               \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                                \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                               \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 4)

#define GET_DATA_BS_FS_YX_BSV16_FSV8_INDEX_SAFE(prefix, b, f, y, x)   \
    get_bs_fs_zyx_bsv_fsv_index_safe(                                \
        b, f, 0, y, x,                                               \
        CAT(prefix, _SIZE_X),                                        \
        CAT(prefix, _SIZE_Y),                                        \
        CAT(prefix, _SIZE_Z),                                        \
        CAT(prefix, _FEATURE_NUM),                                   \
        CAT(prefix, _BATCH_NUM),                                     \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                          \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                           \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                        \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                         \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                             \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 8)

#define GET_DATA_BS_FS_ZYX_BSV16_FSV8_INDEX_SAFE(prefix, b, f, z, y, x) \
    get_bs_fs_zyx_bsv_fsv_index_safe(                                  \
        b, f, z, y, x,                                                 \
        CAT(prefix, _SIZE_X),                                          \
        CAT(prefix, _SIZE_Y),                                          \
        CAT(prefix, _SIZE_Z),                                          \
        CAT(prefix, _FEATURE_NUM),                                     \
        CAT(prefix, _BATCH_NUM),                                       \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                            \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                             \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                          \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                           \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                               \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                                \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                               \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                                \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                               \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 8)

#define GET_DATA_BS_FS_YX_BSV8_FSV4_INDEX_SAFE(prefix, b, f, y, x)   \
    get_bs_fs_zyx_bsv_fsv_index_safe(                                \
        b, f, 0, y, x,                                               \
        CAT(prefix, _SIZE_X),                                        \
        CAT(prefix, _SIZE_Y),                                        \
        CAT(prefix, _SIZE_Z),                                        \
        CAT(prefix, _FEATURE_NUM),                                   \
        CAT(prefix, _BATCH_NUM),                                     \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                          \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                           \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                        \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                         \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                             \
        CAT(prefix, _PAD_AFTER_SIZE_X), 8, 4)

#define GET_DATA_BS_FS_ZYX_BSV8_FSV4_INDEX_SAFE(prefix, b, f, z, y, x) \
    get_bs_fs_zyx_bsv_fsv_index_safe(                                  \
        b, f, z, y, x,                                                 \
        CAT(prefix, _SIZE_X),                                          \
        CAT(prefix, _SIZE_Y),                                          \
        CAT(prefix, _SIZE_Z),                                          \
        CAT(prefix, _FEATURE_NUM),                                     \
        CAT(prefix, _BATCH_NUM),                                       \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                            \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                             \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                          \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                           \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                               \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                                \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                               \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                                \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                               \
        CAT(prefix, _PAD_AFTER_SIZE_X), 8, 4)

#define GET_DATA_BS_FS_YX_BSV16_FSV2_INDEX_SAFE(prefix, b, f, y, x)   \
    get_bs_fs_zyx_bsv_fsv_index_safe(                                \
        b, f, 0, y, x,                                               \
        CAT(prefix, _SIZE_X),                                        \
        CAT(prefix, _SIZE_Y),                                        \
        CAT(prefix, _SIZE_Z),                                        \
        CAT(prefix, _FEATURE_NUM),                                   \
        CAT(prefix, _BATCH_NUM),                                     \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                          \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                           \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                        \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                         \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                             \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 2)

#define GET_DATA_BS_FS_ZYX_BSV16_FSV2_INDEX_SAFE(prefix, b, f, z, y, x) \
    get_bs_fs_zyx_bsv_fsv_index_safe(                                  \
        b, f, z, y, x,                                                 \
        CAT(prefix, _SIZE_X),                                          \
        CAT(prefix, _SIZE_Y),                                          \
        CAT(prefix, _SIZE_Z),                                          \
        CAT(prefix, _FEATURE_NUM),                                     \
        CAT(prefix, _BATCH_NUM),                                       \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                            \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                             \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                          \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                           \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                               \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                                \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                               \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                                \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                               \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 2)

#define GET_DATA_BS_FS_YX_BSV8_FSV2_INDEX_SAFE(prefix, b, f, y, x)   \
    get_bs_fs_zyx_bsv_fsv_index_safe(                                \
        b, f, 0, y, x,                                               \
        CAT(prefix, _SIZE_X),                                        \
        CAT(prefix, _SIZE_Y),                                        \
        CAT(prefix, _SIZE_Z),                                        \
        CAT(prefix, _FEATURE_NUM),                                   \
        CAT(prefix, _BATCH_NUM),                                     \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                          \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                           \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                        \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                         \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                             \
        CAT(prefix, _PAD_AFTER_SIZE_X), 8, 2)

#define GET_DATA_BS_FS_ZYX_BSV8_FSV2_INDEX_SAFE(prefix, b, f, z, y, x) \
    get_bs_fs_zyx_bsv_fsv_index_safe(                                  \
        b, f, z, y, x,                                                 \
        CAT(prefix, _SIZE_X),                                          \
        CAT(prefix, _SIZE_Y),                                          \
        CAT(prefix, _SIZE_Z),                                          \
        CAT(prefix, _FEATURE_NUM),                                     \
        CAT(prefix, _BATCH_NUM),                                       \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                            \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                             \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                          \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                           \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                               \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                                \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                               \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                                \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                               \
        CAT(prefix, _PAD_AFTER_SIZE_X), 8, 2)

#define GET_DATA_BS_FS_YX_BSV4_FSV2_INDEX_SAFE(prefix, b, f, y, x)   \
    get_bs_fs_zyx_bsv_fsv_index_safe(                                \
        b, f, 0, y, x,                                               \
        CAT(prefix, _SIZE_X),                                        \
        CAT(prefix, _SIZE_Y),                                        \
        CAT(prefix, _SIZE_Z),                                        \
        CAT(prefix, _FEATURE_NUM),                                   \
        CAT(prefix, _BATCH_NUM),                                     \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                          \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                           \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                        \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                         \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                             \
        CAT(prefix, _PAD_AFTER_SIZE_X), 4, 2)

#define GET_DATA_BS_FS_ZYX_BSV32_FSV16_INDEX_SAFE(prefix, b, f, z, y, x) \
    get_bs_fs_zyx_bsv_fsv_index_safe(                                    \
        b, f, z, y, x,                                                   \
        CAT(prefix, _SIZE_X),                                            \
        CAT(prefix, _SIZE_Y),                                            \
        CAT(prefix, _SIZE_Z),                                            \
        CAT(prefix, _FEATURE_NUM),                                       \
        CAT(prefix, _BATCH_NUM),                                         \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                              \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                               \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                            \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                                 \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                                  \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                                 \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                                  \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                                 \
        CAT(prefix, _PAD_AFTER_SIZE_X), 32, 16)

#define GET_DATA_BS_FS_YX_BSV32_FSV16_INDEX_SAFE(prefix, b, f, y, x) \
    get_bs_fs_zyx_bsv_fsv_index_safe(                                \
        b, f, 0, y, x,                                               \
        CAT(prefix, _SIZE_X),                                        \
        CAT(prefix, _SIZE_Y),                                        \
        CAT(prefix, _SIZE_Z),                                        \
        CAT(prefix, _FEATURE_NUM),                                   \
        CAT(prefix, _BATCH_NUM),                                     \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                          \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                           \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                        \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                         \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                             \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                              \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                             \
        CAT(prefix, _PAD_AFTER_SIZE_X), 32, 16)

#define GET_DATA_BS_FS_ZYX_BSV16_FSV32_INDEX_SAFE(prefix, b, f, z, y, x) \
    get_bs_fs_zyx_bsv_fsv_index_safe(                                    \
        b, f, z, y, x,                                                   \
        CAT(prefix, _SIZE_X),                                            \
        CAT(prefix, _SIZE_Y),                                            \
        CAT(prefix, _SIZE_Z),                                            \
        CAT(prefix, _FEATURE_NUM),                                       \
        CAT(prefix, _BATCH_NUM),                                         \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                              \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                               \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                            \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                                 \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                                  \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                                 \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                                  \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                                 \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 32)

#define GET_DATA_BS_FS_YX_BSV16_FSV32_INDEX_SAFE(prefix, b, f, y, x) \
    get_bs_fs_zyx_bsv_fsv_index_safe(                                    \
        b, f, 0, y, x,                                                   \
        CAT(prefix, _SIZE_X),                                            \
        CAT(prefix, _SIZE_Y),                                            \
        CAT(prefix, _SIZE_Z),                                            \
        CAT(prefix, _FEATURE_NUM),                                       \
        CAT(prefix, _BATCH_NUM),                                         \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                              \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                               \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                            \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                                 \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                                  \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                                 \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                                  \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                                 \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 32)

#define GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX_SAFE(prefix, b, f, z, y, x) \
    get_bs_fs_zyx_bsv_fsv_index_safe(                                    \
        b, f, z, y, x,                                                   \
        CAT(prefix, _SIZE_X),                                            \
        CAT(prefix, _SIZE_Y),                                            \
        CAT(prefix, _SIZE_Z),                                            \
        CAT(prefix, _FEATURE_NUM),                                       \
        CAT(prefix, _BATCH_NUM),                                         \
        CAT(prefix, _PAD_BEFORE_BATCH_NUM),                              \
        CAT(prefix, _PAD_AFTER_BATCH_NUM),                               \
        CAT(prefix, _PAD_BEFORE_FEATURE_NUM),                            \
        CAT(prefix, _PAD_AFTER_FEATURE_NUM),                             \
        CAT(prefix, _PAD_BEFORE_SIZE_Z),                                 \
        CAT(prefix, _PAD_AFTER_SIZE_Z),                                  \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),                                 \
        CAT(prefix, _PAD_AFTER_SIZE_Y),                                  \
        CAT(prefix, _PAD_BEFORE_SIZE_X),                                 \
        CAT(prefix, _PAD_AFTER_SIZE_X), 16, 16)
