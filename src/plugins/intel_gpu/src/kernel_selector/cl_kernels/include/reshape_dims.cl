// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// ! Huge warning !
// Reshape functions below assume both input and output formats are bfyx, bfzyx, bfwzyx, etc.

inline uint8 FUNC(reshape_dims)(
    uint b, uint f, uint v, uint u, uint w, uint z, uint y, uint x,
    uint src_size_f, uint src_size_v, uint src_size_u, uint src_size_w, uint src_size_z, uint src_size_y, uint src_size_x,
    uint dst_size_f, uint dst_size_v, uint dst_size_u, uint dst_size_w, uint dst_size_z, uint dst_size_y, uint dst_size_x,
    uint src_dims, uint dst_dims)
{
 if (dst_dims == src_dims) {
  return (uint8)(b, f, v, u, w, z, y, x);
 }
 const uint src_pitch_x = 1;
 const uint src_pitch_y = src_pitch_x * src_size_x;
 const uint src_pitch_z = src_pitch_y * src_size_y;
 const uint src_pitch_w = src_pitch_z * src_size_z;
 const uint src_pitch_u = src_pitch_w * src_size_w;
 const uint src_pitch_v = src_pitch_u * src_size_u;
 const uint src_pitch_f = src_pitch_v * src_size_v;
 const uint src_pitch_b = src_pitch_f * src_size_f;
 uint flat_idx = x * src_pitch_x
               + y * src_pitch_y
               + z * src_pitch_z
               + w * src_pitch_w
               + u * src_pitch_u
               + v * src_pitch_v
               + f * src_pitch_f
               + b * src_pitch_b;
 uint dst_x = flat_idx % dst_size_x;
 flat_idx /= dst_size_x;
 uint dst_y = flat_idx % dst_size_y;
 flat_idx /= dst_size_y;
 uint dst_z = flat_idx % dst_size_z;
 flat_idx /= dst_size_z;
 uint dst_w = flat_idx % dst_size_w;
 flat_idx /= dst_size_w;
 uint dst_u = flat_idx % dst_size_u;
 flat_idx /= dst_size_u;
 uint dst_v = flat_idx % dst_size_v;
 flat_idx /= dst_size_v;
 uint dst_f = flat_idx % dst_size_f;
 flat_idx /= dst_size_f;
 uint dst_b = flat_idx;
 return (uint8)(dst_b, dst_f, dst_v, dst_u, dst_w, dst_z, dst_y, dst_x);
}

// ======================================================================
// Reshape dimensions from indices in `src_prefix` DataTensor to indices
// in `dst_prefix` DataTensor.
// Reshaping assumes dimension order: b, f, reverse spatial - w, z, y, x.
#define RESHAPE_DIMS(src_prefix, dst_prefix, b, f, v, u, w, z, y, x)    \
    FUNC_CALL(reshape_dims)(                                            \
        b, f, v, u, w, z, y, x,                                         \
        CAT(src_prefix, _FEATURE_NUM),                                  \
        CAT(src_prefix, _SIZE_V),                                       \
        CAT(src_prefix, _SIZE_U),                                       \
        CAT(src_prefix, _SIZE_W),                                       \
        CAT(src_prefix, _SIZE_Z),                                       \
        CAT(src_prefix, _SIZE_Y),                                       \
        CAT(src_prefix, _SIZE_X),                                       \
        CAT(dst_prefix, _FEATURE_NUM),                                  \
        CAT(dst_prefix, _SIZE_V),                                       \
        CAT(dst_prefix, _SIZE_U),                                       \
        CAT(dst_prefix, _SIZE_W),                                       \
        CAT(dst_prefix, _SIZE_Z),                                       \
        CAT(dst_prefix, _SIZE_Y),                                       \
        CAT(dst_prefix, _SIZE_X),                                       \
        CAT(src_prefix, _DIMS),                                         \
        CAT(dst_prefix, _DIMS))
