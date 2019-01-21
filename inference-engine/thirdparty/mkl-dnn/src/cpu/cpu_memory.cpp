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

#include "format_traits.hpp"

#include "cpu_memory.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl;
using namespace mkldnn::impl::data_type;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;

using dk = data_kind_t;
using bf = block_format_t;

template <data_type_t dt, memory_format_t fmt>
typename utils::enable_if<format_traits<fmt>::data_kind == dk::data>::type
typed_zero_pad_data(
    const memory_desc_wrapper &m_d, typename prec_traits<dt>::type *data) {
    constexpr int blksize = format_traits<fmt>::blk_size;

    const auto &dims = m_d.dims();
    const auto &pdims = m_d.blocking_desc().padding_dims;

    const int C = pdims[1] / blksize - 1;
    const int c_tail_start = dims[1] % blksize;
    assert(c_tail_start != 0);
    const size_t sp_rest = utils::array_product(dims + 3, m_d.ndims() - 3);

    parallel_nd(dims[0], dims[2], [&](int n, int sp0) {
        auto *d = &data[m_d.blk_off(n, C, sp0)];
        for (size_t sp = 0; sp < sp_rest; ++sp) {
            for (int c = c_tail_start; c < blksize; ++c)
                d[sp * blksize + c] = 0;
        }
    });
}

template <data_type_t dt, memory_format_t fmt>
typename utils::enable_if<false
|| format_traits<fmt>::blk_fmt == bf::_8o
|| format_traits<fmt>::blk_fmt == bf::_16o
>::type typed_zero_pad_weights(const memory_desc_wrapper &m_d,
        typename prec_traits<dt>::type *data) {
    static constexpr int w_groups = format_traits<fmt>::data_kind == dk::gwei;
    constexpr int is_1d = format_traits<fmt>::ndims_sp == 1;
    constexpr int is_3d = format_traits<fmt>::ndims_sp == 3;
    constexpr int blksize = format_traits<fmt>::blk_size;

    const auto &dims = m_d.dims();
    const auto &pdims = m_d.blocking_desc().padding_dims;

    const int G = w_groups ? dims[0] : 1;
    const int NB_OC = pdims[w_groups + 0] / blksize;
    const int IC = dims[w_groups + 1];
    const int D = is_3d ? dims[w_groups + 2] : 1;
    const int H = is_1d ? 1 : dims[w_groups + 2 + is_3d];
    const int W = dims[w_groups + 3 - is_1d + is_3d];

    const int oc_tail = pdims[w_groups + 0] - dims[w_groups + 0];

    parallel_nd(G, IC, D, H, W,
        [&](int g, int ic, int d, int h, int w) {
        auto x = &data[wei_blk_off_like_gwei3D<fmt>(m_d,
                g, NB_OC - 1, ic, d, h, w)];
        for (int oc = blksize - oc_tail; oc < blksize; ++oc)
            x[oc] = 0;
    });
}

template <data_type_t dt, memory_format_t fmt>
typename utils::enable_if<false
|| format_traits<fmt>::blk_fmt == bf::_8i
|| format_traits<fmt>::blk_fmt == bf::_16i
>::type typed_zero_pad_weights(const memory_desc_wrapper &m_d,
        typename prec_traits<dt>::type *data) {
    static constexpr int w_groups = format_traits<fmt>::data_kind == dk::gwei;
    constexpr int is_1d = format_traits<fmt>::ndims_sp == 1;
    constexpr int is_3d = format_traits<fmt>::ndims_sp == 3;
    constexpr int blksize = format_traits<fmt>::blk_size;

    const auto &dims = m_d.dims();
    const auto &pdims = m_d.blocking_desc().padding_dims;

    const int G = w_groups ? dims[0] : 1;
    const int OC = dims[w_groups + 0];
    const int NB_IC = pdims[w_groups + 1] / blksize;
    const int D = is_3d ? dims[w_groups + 2] : 1;
    const int H = is_1d ? 1 : dims[w_groups + 2 + is_3d];
    const int W = dims[w_groups + 3 + is_3d];

    const int ic_tail = pdims[w_groups + 1] - dims[w_groups + 1];

    parallel_nd(G, OC, D, H, W,
        [&](int g, int oc, int d, int h, int w) {
        auto x = &data[wei_blk_off_like_gwei3D<fmt>(m_d,
                g, oc, NB_IC - 1, d, h, w)];
        for (int ic = blksize - ic_tail; ic < blksize; ++ic)
            x[ic] = 0;
    });
}

template <data_type_t dt, memory_format_t fmt>
typename utils::enable_if<
block_format_traits<format_traits<fmt>::blk_fmt>::blk_ndims == 2>::type
typed_zero_pad_weights(const memory_desc_wrapper &m_d,
        typename prec_traits<dt>::type *data) {
    using data_t = typename prec_traits<dt>::type;
    static constexpr int w_groups = format_traits<fmt>::data_kind == dk::gwei;
    constexpr int is_1d = format_traits<fmt>::ndims_sp == 1;
    constexpr int is_3d = format_traits<fmt>::ndims_sp == 3;
    constexpr int blksize = format_traits<fmt>::blk_size;
    const auto &dims = m_d.dims();
    const auto &pdims = m_d.blocking_desc().padding_dims;

    const int G = w_groups ? dims[0] : 1;
    const int NB_OC = pdims[w_groups + 0] / blksize;
    const int NB_IC = pdims[w_groups + 1] / blksize;
    const int D = is_3d ? dims[w_groups + 2] : 1;
    const int H = is_1d ? 1 : dims[w_groups + 2 + is_3d];
    const int W = dims[w_groups + 3 - is_1d + is_3d];

    auto ker = [&](data_t *d, const int oc_tail, const int ic_tail) {
#       define blk_off OI_blk_off<format_traits<fmt>::blk_fmt>
        int oc = 0;
        for (; oc < blksize - oc_tail; ++oc) {
            for (int ic = blksize - ic_tail; ic < blksize; ++ic)
                d[blk_off(oc, ic)] = 0;
        }
        for (; oc < blksize; ++oc)
            for (int ic = 0; ic < blksize; ++ic)
                d[blk_off(oc, ic)] = 0;
#       undef blk_off
    };

    const int oc_tail = pdims[w_groups + 0] - dims[w_groups + 0];
    const int ic_tail = pdims[w_groups + 1] - dims[w_groups + 1];

    if (ic_tail) {
        parallel_nd(G, NB_OC, D, H, W,
            [&](int g, int nb_oc, int d, int h, int w) {
            auto x = &data[wei_blk_off_like_gwei3D<fmt>(m_d,
                    g, nb_oc, NB_IC - 1, d, h, w)];
            ker(x, 0, ic_tail);
        });
    }

    if (oc_tail) {
        parallel_nd(G, NB_IC, D, H, W,
            [&](int g, int nb_ic, int d, int h, int w) {
            auto x = &data[wei_blk_off_like_gwei3D<fmt>(m_d,
                    g, NB_OC - 1, nb_ic, d, h, w)];
            ker(x, oc_tail, 0);
        });
    }
}

template <data_type_t dt, memory_format_t fmt>
typename utils::enable_if<false
|| format_traits<fmt>::blk_fmt == bf::_8g
|| format_traits<fmt>::blk_fmt == bf::_16g
>::type typed_zero_pad_weights(const memory_desc_wrapper &m_d,
        typename prec_traits<dt>::type *data) {
    constexpr int blksize = format_traits<fmt>::blk_size;

    const auto &dims = m_d.dims();
    const auto &pdims = m_d.blocking_desc().padding_dims;

    const int G = pdims[0] / blksize - 1;
    const int g_tail_start = dims[0] % blksize;
    assert(g_tail_start != 0);
    const ptrdiff_t sz_rest
        = (ptrdiff_t)utils::array_product(dims + 1, m_d.ndims() - 1);

    auto *d = &data[m_d.blk_off(G)];

    parallel_nd(sz_rest, [&](ptrdiff_t s) {
        for (int g = g_tail_start; g < blksize; ++g)
            d[s * blksize + g] = 0;
    });
}

template <data_type_t dt>
void typed_zero_pad_generic_blocked(const memory_desc_wrapper &m_d,
        typename prec_traits<dt>::type *data) {
    const int ndims = m_d.ndims();
    const auto &dims = m_d.dims();
    const auto &pdims = m_d.blocking_desc().padding_dims;

    const ptrdiff_t nelems = (ptrdiff_t)m_d.nelems(true);

    /* [D_0] .. [D_k][D_k+1] .. [D_ndim - 1]
     *            |  \                     /
     *            |   ---------------------
     *           has        contiguous
     *         padding
     *
     * step     <-- D_k+1 * ... * D_ndims-1
     * step_dim <-- k
     */

    ptrdiff_t step = 1;
    int step_dim = ndims - 1;
    for (; step_dim >= 0; --step_dim) {
        if (dims[step_dim] != pdims[step_dim]) break;
        step *= dims[step_dim];
    }

    assert(step_dim >= 0 && "no zero padding is required");
    if (step_dim < 0) return;

    parallel_nd(nelems, [&](ptrdiff_t e) {
        bool need_zero = false;

        ptrdiff_t idx = e / step;
        for (int d = step_dim; d >= 0; --d) {
            if (idx % pdims[d] >= dims[d]) {
                need_zero = true;
                break;
            }
            idx /= pdims[d];
        }

        if (need_zero) {
            for (ptrdiff_t e0 = 0; e0 < step; ++e0)
                data[m_d.off_l(e + e0, true)] = 0;
        }
    });
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
    MAYBE_DATA(nCw8c);
    MAYBE_DATA(nCw16c);
    MAYBE_DATA(nChw8c);
    MAYBE_DATA(nCdhw8c);
    MAYBE_DATA(nChw16c);
    MAYBE_DATA(nCdhw16c);

    /* weights */
#   define MAYBE_WEIGHTS(f) if (fmt == f) \
    { typed_zero_pad_weights<dt, f>(mpd, data); return success; }
    MAYBE_WEIGHTS(OIdhw8i8o);
    MAYBE_WEIGHTS(OIdhw8o8i);
    MAYBE_WEIGHTS(OIdhw16i16o);
    MAYBE_WEIGHTS(OIdhw16o16i);
    MAYBE_WEIGHTS(Oidhw16o);
    MAYBE_WEIGHTS(Odhwi16o);
    MAYBE_WEIGHTS(Odhwi8o);
    MAYBE_WEIGHTS(oIhw8i);
    MAYBE_WEIGHTS(oIhw16i);
    MAYBE_WEIGHTS(oIdhw8i);
    MAYBE_WEIGHTS(oIdhw16i);
    MAYBE_WEIGHTS(OIhw8i8o);
    MAYBE_WEIGHTS(OIhw16i16o);
    MAYBE_WEIGHTS(OIhw4i16o4i);
    MAYBE_WEIGHTS(OIhw4i16o4i_s8s8);
    MAYBE_WEIGHTS(Owi8o);
    MAYBE_WEIGHTS(OIw8i8o);
    MAYBE_WEIGHTS(OIw8o8i);
    MAYBE_WEIGHTS(OIw16i16o);
    MAYBE_WEIGHTS(OIw16o16i);
    MAYBE_WEIGHTS(Oiw16o);
    MAYBE_WEIGHTS(Owi16o);
    MAYBE_WEIGHTS(OIw8i16o2i);
    MAYBE_WEIGHTS(OIw8o16i2o);
    MAYBE_WEIGHTS(IOw16o16i);
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
    MAYBE_WEIGHTS(gOIhw4i16o4i_s8s8);
    MAYBE_WEIGHTS(gOwi8o);
    MAYBE_WEIGHTS(gOIw8i8o);
    MAYBE_WEIGHTS(gOIw8o8i);
    MAYBE_WEIGHTS(gOIw16i16o);
    MAYBE_WEIGHTS(gOIw16o16i);
    MAYBE_WEIGHTS(gOiw16o);
    MAYBE_WEIGHTS(gOwi16o);
    MAYBE_WEIGHTS(gOIw8i16o2i);
    MAYBE_WEIGHTS(gOIw8o16i2o);
    MAYBE_WEIGHTS(gIOw16o16i);
    MAYBE_WEIGHTS(gOIhw8i16o2i);
    MAYBE_WEIGHTS(gOIdhw8i16o2i);
    MAYBE_WEIGHTS(gOIhw8o16i2o);
    MAYBE_WEIGHTS(gOIhw8o8i);
    MAYBE_WEIGHTS(gOIhw16o16i);
    MAYBE_WEIGHTS(gIOhw16o16i);
    MAYBE_WEIGHTS(gOihw16o);
    MAYBE_WEIGHTS(gOhwi8o);
    MAYBE_WEIGHTS(gOhwi16o);
    MAYBE_WEIGHTS(gOIdhw8i8o);
    MAYBE_WEIGHTS(gOIdhw8o8i);
    MAYBE_WEIGHTS(gOIdhw16i16o);
    MAYBE_WEIGHTS(gOIdhw16o16i);
    MAYBE_WEIGHTS(gOidhw16o);
    MAYBE_WEIGHTS(gOdhwi16o);
    MAYBE_WEIGHTS(gOdhwi8o);
    MAYBE_WEIGHTS(Goihw8g);
    MAYBE_WEIGHTS(Goihw16g);
#   undef MAYBE_WEIGHTS

    // the last line of defence
    if (types::format_normalize(fmt) == blocked) {
        typed_zero_pad_generic_blocked<dt>(mpd, data);
        return success;
    }

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
