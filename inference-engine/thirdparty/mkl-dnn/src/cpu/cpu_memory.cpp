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

#include <assert.h>

#include "memory_pd.hpp"
#include "mkldnn_traits.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_memory.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl;
using namespace mkldnn::impl::data_type;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;

template <data_type_t dt, memory_format_t fmt>
typename utils::enable_if<fmt == nChw8c || fmt == nChw16c || fmt == nCdhw16c
>::type typed_zero_pad_data(const memory_desc_wrapper &m_d,
        typename prec_traits<dt>::type *data) {
    constexpr int blksize = fmt == nChw8c ? 8 : 16;

    const auto &dims = m_d.dims();
    const auto &pdims = m_d.blocking_desc().padding_dims;

    const int C = pdims[1] / blksize - 1;
    const int c_tail_start = dims[1] % blksize;
    assert(c_tail_start != 0);
    const size_t sp_rest = utils::array_product(dims + 3, m_d.ndims() - 3);

#   pragma omp parallel for collapse(2) schedule(static)
    for (int n = 0; n < dims[0]; ++n)
    for (int sp0 = 0; sp0 < dims[2]; ++sp0) {
        auto *d = &data[m_d.blk_off(n, C, sp0)];
        for (size_t sp = 0; sp < sp_rest; ++sp) {
            for (int c = c_tail_start; c < blksize; ++c)
                d[sp * blksize + c] = 0;
        }
    }
}

template <data_type_t dt, memory_format_t fmt>
typename utils::enable_if<false
|| fmt == Ohwi8o || fmt == Oihw16o || fmt == Ohwi16o || fmt == Oidhw16o
|| fmt == Odhwi16o || fmt == gOhwi8o || fmt == gOihw16o || fmt == gOhwi16o
|| fmt == gOidhw16o || fmt == gOdhwi16o
>::type typed_zero_pad_weights(const memory_desc_wrapper &m_d,
        typename prec_traits<dt>::type *data) {
    constexpr int w_groups = false
        || fmt == gOhwi8o || fmt == gOihw16o || fmt == gOhwi16o
        || fmt == gOidhw16o || fmt == gOdhwi16o;

    constexpr int is_3d = false
        || fmt == Oidhw16o || fmt == Odhwi16o
        || fmt == gOidhw16o || fmt == gOdhwi16o;

    constexpr int blksize = fmt == Ohwi8o || fmt == gOhwi8o ? 8 : 16;

    const auto &dims = m_d.dims();
    const auto &pdims = m_d.blocking_desc().padding_dims;

    const int G = w_groups ? dims[0] : 1;
    const int NB_OC = pdims[w_groups + 0] / blksize;
    const int IC = dims[w_groups + 1];
    const int D = is_3d ? dims[w_groups + 2] : 1;
    const int H = dims[w_groups + 2 + is_3d];
    const int W = dims[w_groups + 3 + is_3d];

    const int oc_tail = pdims[w_groups + 0] - dims[w_groups + 0];

#   pragma omp parallel for collapse(5)
    for (int g = 0; g < G; ++g)
    for (int ic = 0; ic < IC; ++ic)
    for (int d = 0; d < D; ++d)
    for (int h = 0; h < H; ++h)
    for (int w = 0; w < W; ++w) {
        auto x = &data[is_3d
            ? m_d.blk_off<!w_groups>(g, NB_OC - 1, ic, d, h, w)
            : m_d.blk_off<!w_groups>(g, NB_OC - 1, ic, h, w) ];
        for (int oc = blksize - oc_tail; oc < blksize; ++oc)
            x[oc] = 0;
    }
}

template <data_type_t dt, memory_format_t fmt>
typename utils::enable_if<fmt == oIhw8i || fmt == oIhw16i>::type
typed_zero_pad_weights(const memory_desc_wrapper &m_d,
        typename prec_traits<dt>::type *data) {
    constexpr int blksize = fmt == oIhw8i ? 8 : 16;

    constexpr int w_groups = 0;
    constexpr int is_3d = 0;

    const auto &dims = m_d.dims();
    const auto &pdims = m_d.blocking_desc().padding_dims;

    const int G = w_groups ? dims[0] : 1;
    const int OC = dims[w_groups + 0];
    const int NB_IC = pdims[w_groups + 1] / blksize;
    const int D = is_3d ? dims[w_groups + 2] : 1;
    const int H = dims[w_groups + 2 + is_3d];
    const int W = dims[w_groups + 3 + is_3d];

    const int ic_tail = pdims[w_groups + 1] - dims[w_groups + 1];

#   pragma omp parallel for collapse(5)
    for (int g = 0; g < G; ++g)
    for (int oc = 0; oc < OC; ++oc)
    for (int d = 0; d < D; ++d)
    for (int h = 0; h < H; ++h)
    for (int w = 0; w < W; ++w) {
        auto x = &data[is_3d
            ? m_d.blk_off<!w_groups>(g, oc, NB_IC - 1, d, h, w)
            : m_d.blk_off<!w_groups>(g, oc, NB_IC - 1, h, w) ];
        for (int ic = blksize - ic_tail; ic < blksize; ++ic)
            x[ic] = 0;
    }
}

template <data_type_t dt, memory_format_t fmt>
typename utils::enable_if<false
|| fmt == IOhw16o16i || fmt == gIOhw16o16i
|| fmt == OIdhw16i16o || fmt == OIdhw16o16i || fmt == OIhw8i8o
|| fmt == OIhw16i16o || fmt == OIhw4i16o4i || fmt == OIhw8i16o2i
|| fmt == OIdhw8i16o2i || fmt == OIhw8o16i2o || fmt == OIhw8o8i
|| fmt == OIhw16o16i || fmt == gOIhw8i8o
|| fmt == gOIhw16i16o || fmt == gOIhw4i16o4i || fmt == gOIhw8i16o2i
|| fmt == gOIdhw8i16o2i || fmt == gOIhw8o16i2o || fmt == gOIhw8o8i
|| fmt == gOIhw16o16i || fmt == gOIdhw16i16o || fmt == gOIdhw16o16i
>::type typed_zero_pad_weights(const memory_desc_wrapper &m_d,
        typename prec_traits<dt>::type *data) {
    using data_t = typename prec_traits<dt>::type;
    constexpr int w_groups = false
        || fmt == gOIhw8i8o || fmt == gOIhw16i16o || fmt == gOIhw4i16o4i
        || fmt == gOIhw8i16o2i || fmt == gOIdhw8i16o2i || fmt == gOIhw8o16i2o
        || fmt == gOIhw8o8i || fmt == gOIhw16o16i || fmt == gIOhw16o16i
        || fmt == gOIdhw16i16o || fmt == gOIdhw16o16i || fmt == gOidhw16o
        || fmt == gOdhwi16o;

    constexpr int is_3d = false
        || fmt == OIdhw16i16o || fmt == OIdhw16o16i || fmt == OIdhw8i16o2i
        || fmt == gOIdhw8i16o2i || fmt == gOIdhw16i16o || fmt == gOIdhw16o16i;

    constexpr int blksize = 16;

    const auto &dims = m_d.dims();
    const auto &pdims = m_d.blocking_desc().padding_dims;

    const int G = w_groups ? dims[0] : 1;
    const int NB_OC = pdims[w_groups + 0] / blksize;
    const int NB_IC = pdims[w_groups + 1] / blksize;
    const int D = is_3d ? dims[w_groups + 2] : 1;
    const int H = dims[w_groups + 2 + is_3d];
    const int W = dims[w_groups + 3 + is_3d];

    auto index = [&](const int ic, const int oc) {
        if (utils::one_of(fmt,
                    OIhw8i16o2i, gOIhw8i16o2i,
                    OIdhw8i16o2i, gOIdhw8i16o2i))
            return ((ic / 2) * blksize * 2 + 2 * oc + ic % 2);
        else if (utils::one_of(fmt, OIhw4i16o4i, gOIhw4i16o4i))
            return ((ic / 4) * blksize * 4 + oc * 4 + ic % 4);
        else if (utils::one_of(fmt, OIhw8o16i2o, gOIhw8o16i2o))
            return ((oc / 2) * blksize * 2 + 2 * ic + oc % 2);
        else if (utils::one_of(fmt,
                    OIhw16i16o, gOIhw16i16o,
                    OIdhw16i16o, gOIdhw16i16o))
            return (ic * blksize + oc);
        else
            return (oc * blksize + ic);
    };

    auto ker = [&](data_t *d, const int oc_tail, const int ic_tail) {
        int oc = 0;
        for (; oc < blksize - oc_tail; ++oc) {
            for (int ic = blksize - ic_tail; ic < blksize; ++ic)
                d[index(ic, oc)] = 0;
        }
        for (; oc < blksize; ++oc)
            for (int ic = 0; ic < blksize; ++ic)
                d[index(ic, oc)] = 0;
    };

    const int oc_tail = pdims[w_groups + 0] - dims[w_groups + 0];
    const int ic_tail = pdims[w_groups + 1] - dims[w_groups + 1];

    if (ic_tail) {
#       pragma omp parallel for collapse(5)
        for (int g = 0; g < G; ++g)
        for (int nb_oc = 0; nb_oc < NB_OC; ++nb_oc)
        for (int d = 0; d < D; ++d)
        for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            auto x = &data[is_3d
                ? m_d.blk_off<!w_groups>(g, nb_oc, NB_IC - 1, d, h, w)
                : m_d.blk_off<!w_groups>(g, nb_oc, NB_IC - 1, h, w) ];
            ker(x, 0, ic_tail);
        }
    }

    if (oc_tail) {
#       pragma omp parallel for collapse(5)
        for (int g = 0; g < G; ++g)
        for (int nb_ic = 0; nb_ic < NB_IC; ++nb_ic)
        for (int d = 0; d < D; ++d)
        for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            auto x = &data[is_3d
                ? m_d.blk_off<!w_groups>(g, NB_OC - 1, nb_ic, d, h, w)
                : m_d.blk_off<!w_groups>(g, NB_OC - 1, nb_ic, h, w) ];
            ker(x, oc_tail, 0);
        }
    }
}

template <data_type_t dt, memory_format_t fmt>
typename utils::enable_if<fmt == Goihw8g || fmt == Goihw16g>::type
typed_zero_pad_weights(const memory_desc_wrapper &m_d,
        typename prec_traits<dt>::type *data) {
    constexpr int blksize = fmt == Goihw8g ? 8 : 16;

    const auto &dims = m_d.dims();
    const auto &pdims = m_d.blocking_desc().padding_dims;

    const int G = pdims[0] / blksize - 1;
    const int g_tail_start = dims[0] % blksize;
    assert(g_tail_start != 0);
    const ptrdiff_t sz_rest
        = (ptrdiff_t)utils::array_product(dims + 1, m_d.ndims() - 1);

    auto *d = &data[m_d.blk_off(G)];

#   pragma omp parallel for schedule(static)
    for (ptrdiff_t s = 0; s < sz_rest; ++s) {
        for (int g = g_tail_start; g < blksize; ++g)
            d[s * blksize + g] = 0;
    }
}

template <data_type_t dt>
status_t cpu_memory_t::typed_zero_pad() {
    const memory_desc_wrapper mpd(&conf_);

    // FIXME: guard this check for non-blocked layout
    if (mpd.nelems(false) == mpd.nelems(true))
        return success;

    auto *data = (typename prec_traits<dt>::type *)data_;
    const auto fmt = mpd.format();

    /* data */
#   define MAYBE_DATA(f) if (fmt == f) \
    { typed_zero_pad_data<dt, f>(mpd, data); return success; }
    MAYBE_DATA(nChw8c);
    MAYBE_DATA(nChw16c);
    MAYBE_DATA(nCdhw16c);

    /* weights */
#   define MAYBE_WEIGHTS(f) if (fmt == f) \
    { typed_zero_pad_weights<dt, f>(mpd, data); return success; }
    MAYBE_WEIGHTS(OIdhw16i16o);
    MAYBE_WEIGHTS(OIdhw16o16i);
    MAYBE_WEIGHTS(Oidhw16o);
    MAYBE_WEIGHTS(Odhwi16o);
    MAYBE_WEIGHTS(oIhw8i);
    MAYBE_WEIGHTS(oIhw16i);
    MAYBE_WEIGHTS(OIhw8i8o);
    MAYBE_WEIGHTS(OIhw16i16o);
    MAYBE_WEIGHTS(OIhw4i16o4i);
    MAYBE_WEIGHTS(OIhw8i16o2i);
    MAYBE_WEIGHTS(OIdhw8i16o2i);
    MAYBE_WEIGHTS(OIhw8o16i2o);
    MAYBE_WEIGHTS(OIhw8o8i);
    MAYBE_WEIGHTS(OIhw16o16i);
    MAYBE_WEIGHTS(IOhw16o16i);
    MAYBE_WEIGHTS(Oihw16o);
    MAYBE_WEIGHTS(Ohwi8o);
    MAYBE_WEIGHTS(Ohwi16o);
    MAYBE_WEIGHTS(gOIhw8i8o);
    MAYBE_WEIGHTS(gOIhw16i16o);
    MAYBE_WEIGHTS(gOIhw4i16o4i);
    MAYBE_WEIGHTS(gOIhw8i16o2i);
    MAYBE_WEIGHTS(gOIdhw8i16o2i);
    MAYBE_WEIGHTS(gOIhw8o16i2o);
    MAYBE_WEIGHTS(gOIhw8o8i);
    MAYBE_WEIGHTS(gOIhw16o16i);
    MAYBE_WEIGHTS(gIOhw16o16i);
    MAYBE_WEIGHTS(gOihw16o);
    MAYBE_WEIGHTS(gOhwi8o);
    MAYBE_WEIGHTS(gOhwi16o);
    MAYBE_WEIGHTS(gOIdhw16i16o);
    MAYBE_WEIGHTS(gOIdhw16o16i);
    MAYBE_WEIGHTS(gOidhw16o);
    MAYBE_WEIGHTS(gOdhwi16o);
    MAYBE_WEIGHTS(Goihw8g);
    MAYBE_WEIGHTS(Goihw16g);
#   undef MAYBE_WEIGHTS

    return unimplemented;
}

status_t cpu_memory_t::zero_pad() {
    memory_desc_wrapper md(&conf_);
    const bool skip_zeroing = false
        || data_ == nullptr
        || md.is_zero()
        || !md.is_blocking_desc();
    if (skip_zeroing) return success;

    switch (md.data_type()) {
        case f32: return typed_zero_pad<f32>();
        case s32: return typed_zero_pad<s32>();
        case s16: return typed_zero_pad<s16>();
        case s8: return typed_zero_pad<s8>();
        case u8: return typed_zero_pad<u8>();
        default: assert(!"memory is undefined"); return unimplemented;
    }
    return unimplemented;
}

}
}
}
