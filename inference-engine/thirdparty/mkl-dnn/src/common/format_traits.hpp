/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef FORMAT_TRAITS_HPP
#define FORMAT_TRAITS_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {

enum class data_kind_t {
    data,
    wei,
    gwei,
    rnn,
    other,
};

enum class block_format_t {
    _,
    _4c, _4i, _4o,
    _8c, _8g, _8i, _8o,
    _4i4o, _4o4i, _4o4i_s8s8,
    _8i8o, _8o8i,
    _8o4i, _8o4i_s8s8,
    _8o32i, _16o32i,
    _16c, _16g, _16g_s8s8, _16i, _16o,
    _16i16o, _16o16i,
    _8i16o2i, _8o16i2o,
    _4i16o4i, _4i16o4i_s8s8,
    _2i8o4i, _2i8o4i_s8s8
};

template <block_format_t f> struct block_format_traits {
    using bf = block_format_t;
    static constexpr int levels = f == bf::_
        ? 0
        : utils::one_of(f, bf::_8i16o2i, bf::_8o16i2o,
                           bf::_4i16o4i, bf::_4i16o4i_s8s8,
                           bf::_2i8o4i, bf::_2i8o4i_s8s8) ? 2 : 1;
    static constexpr int blk_ndims = f == bf::_
        ? 0
        : utils::one_of(f, bf::_4c, bf::_4i, bf::_4o, bf::_8c, bf::_8g, bf::_8i, bf::_8o, bf::_16c,
                bf::_16g, bf::_16g_s8s8, bf::_16i, bf::_16o) ? 1 : 2;
    static constexpr int blk_size = f == bf::_
        ? 1
        : (utils::one_of(f, bf::_4c, bf::_4i, bf::_4o, bf::_4i4o, bf::_4o4i, bf::_4o4i_s8s8) ? 4
                : (utils::one_of(f, bf::_8c, bf::_8g, bf::_8i, bf::_8o,
                        bf::_8i8o, bf::_8o8i,
                        bf::_8o4i, bf::_8o4i_s8s8,
                        bf::_2i8o4i, bf::_2i8o4i_s8s8,
                        bf::_8o32i) ? 8 : 16));
};

template <memory_format_t> struct format_traits {
    // data_kind_t data_kind;   -- the kind of data (e.g. weights or rnn)
    // block_format_t blk_fmt;  -- the format of blocks (e.g. 8c or 4i16o4i)
    // int ndims;               -- # of dimensions
    // int ndims_sp;            -- # of spatial dimensions
    // int blk_size;            -- block size (1, 4, 8, or 16)
};

#define DECL_TRAITS(_fmt, _data_kind, _blk_fmt, _ndims, _ndims_sp) \
template <> struct format_traits<memory_format::_fmt> { \
    static constexpr data_kind_t data_kind = data_kind_t::_data_kind; \
    static constexpr block_format_t blk_fmt = block_format_t::_blk_fmt; \
    static constexpr int ndims = _ndims; \
    static constexpr int ndims_sp = _ndims_sp; \
    static constexpr int blk_size = \
        block_format_traits<block_format_t::_blk_fmt>::blk_size; \
}

DECL_TRAITS(any, other, _, 0, 0);
DECL_TRAITS(blocked, other, _, 0, 0);
DECL_TRAITS(x, other, _, 1, 1);

/* data: 2D */
DECL_TRAITS(nc, data, _, 2, 0);

/* data: 3D */
DECL_TRAITS(ncw, data, _, 3, 1);
DECL_TRAITS(nwc, data, _, 3, 1);
DECL_TRAITS(nCw4c, data, _4c, 3, 1);
DECL_TRAITS(nCw8c, data, _8c, 3, 1);
DECL_TRAITS(nCw16c, data, _16c, 3, 1);

/* data: 4D */
DECL_TRAITS(nchw, data, _, 4, 2);
DECL_TRAITS(nhwc, data, _, 4, 2);
DECL_TRAITS(chwn, data, _, 4, 2);
DECL_TRAITS(nChw4c, data, _4c, 4, 2);
DECL_TRAITS(nChw8c, data, _8c, 4, 2);
DECL_TRAITS(nChw16c, data, _16c, 4, 2);

/* data: 5D */
DECL_TRAITS(ncdhw, data, _, 5, 3);
DECL_TRAITS(ndhwc, data, _, 5, 3);
DECL_TRAITS(nCdhw4c, data, _4c, 5, 3);
DECL_TRAITS(nCdhw8c, data, _8c, 5, 3);
DECL_TRAITS(nCdhw16c, data, _16c, 5, 3);

/* wei: 2D */
DECL_TRAITS(oi, wei, _, 2, 0);
DECL_TRAITS(io, wei, _, 2, 0);

/* wei: 3D */
DECL_TRAITS(oiw, wei, _, 3, 1);
DECL_TRAITS(wio, wei, _, 3, 1);
DECL_TRAITS(Owi4o, wei, _4o, 3, 1);
DECL_TRAITS(OIw4i4o, wei, _4i4o, 3, 1);
DECL_TRAITS(Owi8o, wei, _8o, 3, 1);
DECL_TRAITS(OIw8i8o, wei, _8i8o, 3, 1);
DECL_TRAITS(OIw8o8i, wei, _8o8i, 3, 1);
DECL_TRAITS(OIw16i16o, wei, _16i16o, 3, 1);
DECL_TRAITS(OIw16o16i, wei, _16o16i, 3, 1);
DECL_TRAITS(Oiw4o, wei, _4o, 3, 1);
DECL_TRAITS(Oiw16o, wei, _16o, 3, 1);
DECL_TRAITS(Owi16o, wei, _16o, 3, 1);
DECL_TRAITS(OIw8i16o2i, wei, _8i16o2i, 3, 1);
DECL_TRAITS(IOw16o16i, wei, _16o16i, 3, 1);
DECL_TRAITS(OIw8o16i2o, wei, _8o16i2o, 3, 1);

/* wei: 4D */
DECL_TRAITS(oihw, wei, _, 4, 2);
DECL_TRAITS(ihwo, wei, _, 4, 2);
DECL_TRAITS(hwio, wei, _, 4, 2);
DECL_TRAITS(iohw, wei, _, 4, 2);
DECL_TRAITS(hwio_s8s8, wei, _, 4, 2);
DECL_TRAITS(oIhw8i, wei, _8i, 4, 2);
DECL_TRAITS(oIhw16i, wei, _16i, 4, 2);
DECL_TRAITS(OIhw4i4o, wei, _4i4o, 4, 2);
DECL_TRAITS(OIhw8i8o, wei, _8i8o, 4, 2);
DECL_TRAITS(OhIw8o32i, wei, _8o32i, 4, 2);
DECL_TRAITS(OhIw16o32i, wei, _16o32i, 4, 2);
DECL_TRAITS(OhIw8o4i, wei, _8o4i, 4, 2);
DECL_TRAITS(OhIw8o4i_s8s8, wei, _8o4i_s8s8, 4, 2);
DECL_TRAITS(OIhw16i16o, wei, _16i16o, 4, 2);
DECL_TRAITS(OIhw4i16o4i, wei, _4i16o4i, 4, 2);
DECL_TRAITS(OIhw4i16o4i_s8s8, wei, _4i16o4i_s8s8, 4, 2);
DECL_TRAITS(OIhw8i16o2i, wei, _8i16o2i, 4, 2);
DECL_TRAITS(OIhw8o16i2o, wei, _8o16i2o, 4, 2);
DECL_TRAITS(OIhw8o8i, wei, _8o8i, 4, 2);
DECL_TRAITS(OIhw16o16i, wei, _16o16i, 4, 2);
DECL_TRAITS(IOhw16o16i, wei, _16o16i, 4, 2);
DECL_TRAITS(Oihw4o, wei, _4o, 4, 2);
DECL_TRAITS(Oihw16o, wei, _16o, 4, 2);
DECL_TRAITS(Ohwi8o, wei, _8o, 4, 2);
DECL_TRAITS(Ohwi4o, wei, _4o, 4, 2);
DECL_TRAITS(Ohwi16o, wei, _16o, 4, 2);

/* wei: 5D */
DECL_TRAITS(dhwio, wei, _, 5, 3);
DECL_TRAITS(oidhw, wei, _, 5, 3);
DECL_TRAITS(OIdhw4i4o, wei, _4i4o, 5, 3);
DECL_TRAITS(Odhwi4o, wei, _4o, 5, 3);
DECL_TRAITS(OIdhw8i8o, wei, _8i8o, 5, 3);
DECL_TRAITS(OIdhw8o8i, wei, _8o8i, 5, 3);
DECL_TRAITS(Odhwi8o, wei, _8o, 5, 3);
DECL_TRAITS(OIdhw16i16o, wei, _16i16o, 5, 3);
DECL_TRAITS(OIdhw16o16i, wei, _16o16i, 5, 3);
DECL_TRAITS(Oidhw4o, wei, _4o, 5, 3);
DECL_TRAITS(Oidhw16o, wei, _16o, 5, 3);
DECL_TRAITS(Odhwi16o, wei, _16o, 5, 3);
DECL_TRAITS(oIdhw8i, wei, _8i, 5, 3);
DECL_TRAITS(oIdhw16i, wei, _16i, 5, 3);
DECL_TRAITS(OIdhw8i16o2i, wei, _8i16o2i, 5, 3);

/* gwei: 4D */
DECL_TRAITS(goiw, gwei, _, 4, 1);
DECL_TRAITS(gOwi4o, gwei, _4o, 4, 1);
DECL_TRAITS(gOIw4i4o, gwei, _4i4o, 4, 1);
DECL_TRAITS(gOwi8o, gwei, _8o, 4, 1);
DECL_TRAITS(gOIw8i8o, gwei, _8i8o, 4, 1);
DECL_TRAITS(gOIw8o8i, gwei, _8o8i, 4, 1);
DECL_TRAITS(gOIw16i16o, gwei, _16i16o, 4, 1);
DECL_TRAITS(gOIw16o16i, gwei, _16o16i, 4, 1);
DECL_TRAITS(gOiw4o, gwei, _4o, 4, 1);
DECL_TRAITS(gOiw16o, gwei, _16o, 4, 1);
DECL_TRAITS(gOwi16o, gwei, _16o, 4, 1);
DECL_TRAITS(gOIw8i16o2i, gwei, _8i16o2i, 4, 1);
DECL_TRAITS(gIOw16o16i, gwei, _16o16i, 4, 1);
DECL_TRAITS(gOIw8o16i2o, gwei, _8o16i2o, 4, 1);

/* gwei: 5D */
DECL_TRAITS(goihw, gwei, _, 5, 2);
DECL_TRAITS(hwigo, gwei, _, 5, 2);
DECL_TRAITS(giohw, gwei, _, 5, 2);
DECL_TRAITS(hwigo_s8s8, gwei, _, 5, 2);
DECL_TRAITS(gOIhw4i4o, gwei, _4i4o, 5, 2);
DECL_TRAITS(gOIhw8i8o, gwei, _8i8o, 5, 2);
DECL_TRAITS(gOhIw8o4i, gwei, _8o4i, 5, 2);
DECL_TRAITS(gOhIw8o4i_s8s8, gwei, _8o4i_s8s8, 5, 2);
DECL_TRAITS(gOIhw16i16o, gwei, _16i16o, 5, 2);
DECL_TRAITS(gOIhw4i16o4i, gwei, _4i16o4i, 5, 2);
DECL_TRAITS(gOIhw4i16o4i_s8s8, gwei, _4i16o4i_s8s8, 5, 2);
DECL_TRAITS(gOIhw2i8o4i, gwei, _2i8o4i, 5, 2);
DECL_TRAITS(gOIhw2i8o4i_s8s8, gwei, _2i8o4i_s8s8, 5, 2);
DECL_TRAITS(gOIhw8i16o2i, gwei, _8i16o2i, 5, 2);
DECL_TRAITS(gOIdhw8i16o2i, gwei, _8i16o2i, 5, 2);
DECL_TRAITS(gOIhw8o16i2o, gwei, _8o16i2o, 5, 2);
DECL_TRAITS(gOIhw8o8i, gwei, _8o8i, 5, 2);
DECL_TRAITS(gOIhw4o4i, gwei, _4o4i, 5, 2);
DECL_TRAITS(gOIhw4o4i_s8s8, gwei, _4o4i_s8s8, 5, 2);
DECL_TRAITS(gOIhw16o16i, gwei, _16o16i, 5, 2);
DECL_TRAITS(gIOhw16o16i, gwei, _16o16i, 5, 2);
DECL_TRAITS(gOihw4o, gwei, _4o, 5, 2);
DECL_TRAITS(gOihw16o, gwei, _16o, 5, 2);
DECL_TRAITS(gOhwi8o, gwei, _8o, 5, 2);
DECL_TRAITS(gOhwi4o, gwei, _4o, 5, 2);
DECL_TRAITS(gOhwi16o, gwei, _16o, 5, 2);
DECL_TRAITS(Goihw8g, gwei, _8g, 5, 2);
DECL_TRAITS(Goihw16g, gwei, _16g, 5, 2);
DECL_TRAITS(Goihw16g_s8s8, gwei, _16g_s8s8, 5, 2);

/* gwei: 6D */
DECL_TRAITS(goidhw, gwei, _, 6, 3);
DECL_TRAITS(gOIdhw4i4o, gwei, _4i4o, 6, 3);
DECL_TRAITS(gOIdhw8i8o, gwei, _8i8o, 6, 3);
DECL_TRAITS(gOIdhw8o8i, gwei, _8o8i, 6, 3);
DECL_TRAITS(gOdhwi8o, gwei, _8o, 6, 3);
DECL_TRAITS(gOIdhw16i16o, gwei, _16i16o, 6, 3);
DECL_TRAITS(gOIdhw16o16i, gwei, _16o16i, 6, 3);
DECL_TRAITS(gOidhw4o, gwei, _4o, 6, 3);
DECL_TRAITS(gOidhw16o, gwei, _16o, 6, 3);
DECL_TRAITS(gOdhwi16o, gwei, _16o, 6, 3);

/* rnn */
DECL_TRAITS(ntc, rnn, _, 3, 0);
DECL_TRAITS(tnc, rnn, _, 3, 0);
DECL_TRAITS(ldsnc, rnn, _, 5, 0);
DECL_TRAITS(ldigo, rnn, _, 5, 0);
DECL_TRAITS(ldgoi, rnn, _, 5, 0);
DECL_TRAITS(ldgo, rnn, _, 4, 0);

#undef DECL_TRAITS

/** returns the offset within the block for weights blocked over oc and ic */
template <block_format_t f>
constexpr int OI_blk_off(int oc, int ic) {
    using bf = block_format_t;
    static_assert(utils::one_of(f, bf::_4i4o, bf::_4o4i, bf::_4o4i_s8s8,
                bf::_8i8o, bf::_8o8i, bf::_16i16o,
                bf::_16o16i, bf::_8i16o2i, bf::_8o16i2o,
                bf::_4i16o4i, bf::_4i16o4i_s8s8,
                bf::_2i8o4i, bf::_2i8o4i_s8s8,
                bf::_8o4i, bf::_8o4i_s8s8,
                bf::_8o32i, bf::_16o32i),
            "unexpected blocked format");
#   define blksize block_format_traits<f>::blk_size
    return f == bf::_8i16o2i
        ? (ic / 2) * blksize * 2 + 2 * oc + ic % 2
        : (f == bf::_4i16o4i || f == bf::_4i16o4i_s8s8
        || f == bf::_2i8o4i || f == bf::_2i8o4i_s8s8)
        ? (ic / 4) * blksize * 4 + oc * 4 + ic % 4
        : f == bf::_8o16i2o
        ? (oc / 2) * blksize * 2 + 2 * ic + oc % 2
        : utils::one_of(f, bf::_4i4o, bf::_8i8o, bf::_16i16o)
        ? ic * blksize + oc
        : (f == bf::_8o4i || f == bf::_8o4i_s8s8)
        ? (ic / 4) * blksize * 4 + 4 * oc + ic % 4
        : (f == bf::_8o32i || f == bf::_16o32i)
        ? 32 * oc + 32
        : oc * blksize + ic;
#   undef blksize // if only we program in C++14...
}

/** computes offset for 1D, 2D, or 3D weights (w/ or w/o groups)
 * in the same fashion: off(g, oc, ic, d, h, w) */
template <memory_format_t fmt>
constexpr size_t wei_blk_off_like_gwei3D(const memory_desc_wrapper &md,
        const int g, const int o, const int i, const int d, const int h,
        const int w) {
    static_assert(utils::one_of(format_traits<fmt>::data_kind,
                data_kind_t::wei, data_kind_t::gwei), "weights are expected");
    static_assert(utils::one_of(format_traits<fmt>::ndims_sp, 1, 2, 3),
            "incorrect number of dims");
#   define w_grp (format_traits<fmt>::data_kind == data_kind_t::gwei)
    return format_traits<fmt>::ndims_sp == 1
        ? md.blk_off<!w_grp>(g, o, i, w)
        : format_traits<fmt>::ndims_sp == 2
            ? md.blk_off<!w_grp>(g, o, i, h, w)
            : md.blk_off<!w_grp>(g, o, i, d, h, w);
#   undef w_grp // if only we program in C++14...
}

} // namespace impl
} // namespace mkldnn

#endif
