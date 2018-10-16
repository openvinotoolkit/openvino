/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef MEMORY_DESC_WRAPPER_HPP
#define MEMORY_DESC_WRAPPER_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "utils.hpp"

#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {

/** thin wrapper class over \struct memory_desc_t which allows easy
 * manipulatings with underlying C structure, which is taken by refernce */
struct memory_desc_wrapper: public c_compatible {
    const memory_desc_t *_md;

    /** constructor which takes a reference to a constant underlying C memory
     * descriptor \param md */
    memory_desc_wrapper(const memory_desc_t &md) : _md(&md) {}
    memory_desc_wrapper(const memory_desc_t *md) : _md(md) {}
    memory_desc_wrapper(const memory_pd_t *m_pd);

    /* implementing attrubutes */
    inline int ndims() const { return _md->ndims; }
    const dims_t &dims() const { return _md->dims; }
    data_type_t data_type() const { return _md->data_type; }
    memory_format_t format() const { return _md->format; }
    bool is_blocking_desc() const {
        return (format() != memory_format::wino_fmt
                && format() != memory_format::any
                && format() != memory_format::undef);
    }
    bool is_wino_desc() const {
        return (format() == memory_format::wino_fmt);
    }
    const blocking_desc_t &blocking_desc() const {
        assert(is_blocking_desc());
        return _md->layout_desc.blocking;
    }
    const wino_data_t &wino_desc() const {
        assert(is_wino_desc());
        return _md->layout_desc.wino_desc;
    }

    /* some useful function */

    /** returns the number of elements including padding if \param with_padding
     * is true, and the number of data elements otherwise */
    size_t nelems(bool with_padding = false) const {
        if (is_zero()) return 0;
    return utils::array_product<int, size_t>(with_padding
                ? blocking_desc().padding_dims : dims(), ndims());
    }

    /** returns true if memory descriptor is zero */
    bool is_zero() const { return ndims() == 0; }

    /** return the size of data type (a shortcut) */
    size_t data_type_size() const
    { return types::data_type_size(data_type()); }

    /** returns the size required to store described memory
     * note: if offset_padding != 0 returns 0 (need to specify the behavior) */
    size_t size() const {
        using namespace mkldnn::impl::memory_format;
        if (is_zero() || format() == memory_format::any) return 0;

        assert(utils::one_of(format(), blocked, x, nc, nchw, nhwc, chwn,
                    nChw8c, nChw16c, oi, io, oihw, ihwo, hwio, hwigo, oIhw8i,
                    oIhw16i, OIhw8i8o, OIhw16i16o, OIhw8i16o2i, OIhw8o16i2o,
                    OIhw8o8i, OIhw16o16i, Oihw16o, Ohwi8o, Ohwi16o,
                    OIhw4i16o4i, goihw, gOIhw8i8o, gOIhw16i16o, gOIhw8i16o2i,
                    gOIhw8o16i2o, gOIhw8o8i, gOIhw16o16i, gOihw16o, gOhwi8o,
                    gOhwi16o, IOhw16o16i, gIOhw16o16i, gOIhw4i16o4i, Goihw8g,
                    Goihw16g, ncdhw, oidhw, goidhw, nCdhw16c, OIdhw16i16o,
                    gOIdhw16i16o, OIdhw16o16i, gOIdhw16o16i, ndhwc, gOidhw16o,
                    Oidhw16o, gOdhwi16o, Odhwi16o, ntc, tnc, ldsnc, ldigo,
                    ldgoi, ldgo, wino_fmt, dhwio, OIdhw8i16o2i, gOIdhw8i16o2i));
        if (format() == wino_fmt) {
            return wino_desc().size;
        } else {
            if (blocking_desc().offset_padding != 0) return 0;

            const auto &block_dims = blocking_desc().block_dims;
            const auto &strides = blocking_desc().strides;
            const auto &padding_dims = blocking_desc().padding_dims;

            size_t max_size = 0;
            for (int d = 0; d < ndims(); ++d) {
                auto block = block_dims[d];
                max_size = nstl::max(max_size,
                    size_t(padding_dims[d] / block)*strides[0][d]);
                if (block > 1)
                    max_size = nstl::max(max_size, size_t(block*strides[1][d]));
            }
            return max_size * data_type_size();
        }
    }

    /** returns true if data is dense in memory */
    bool is_dense(bool with_padding = false) const;

    /** returns true if memory desc is fully defined */
    bool is_defined() const { return format() != memory_format::any; }

    /** returns true if the only (potentially) padded dim is \param dim */
    bool only_padded_dim(int dim) const {
        assert(is_blocking_desc());
        const auto pdims = blocking_desc().padding_dims;
        for (int d = 0; d < ndims(); ++d)
            if (d != dim && dims()[d] != pdims[d])
                return false;
        return true;
    }

    /* comparison section */

    inline bool operator==(const memory_desc_wrapper &rhs) const;
    inline bool operator!=(const memory_desc_wrapper &rhs) const
    { return !operator==(rhs); }
    inline bool operator==(const memory_desc_t &rhs) const
    { return operator==(memory_desc_wrapper(rhs)); }
    inline bool operator!=(const memory_desc_t &rhs) const
    { return !operator==(rhs); }

    /** returns true if data (w/o padding if with_padding == false and w/
     * padding otherwise) have the same physical structure, i.e. dimensions,
     * strides, and blocked structure. depending on with_data_type flag
     * data_type is taken or not taken into account. dim_start allows to chech
     * similarity for the logical part of data [dim_start .. ndims()].
     * CAUTION: format any and undef are not similiar to whatever, hence the
     * following statement might be true: lhs == rhs && !lhs.similar_to(rhs) */
    /* TODO: revise */
    inline bool similar_to(const memory_desc_wrapper &rhs,
            bool with_padding = true, bool with_data_type = true,
            int dim_start = 0) const;

    /** returns true if one memory can be reordered to another */
    inline bool consistent_with(const memory_desc_wrapper &rhs) const;

    /* offset section */

    /** returns physical offset by logical one. logical offset is represented by
     * an array \param pos. if \param is_pos_padded is true \param pos
     * represents the position in already padded area */
    inline size_t off_v(const dims_t pos, bool is_pos_padded = false) const {
        using namespace mkldnn::impl::memory_format;
        assert(format() != memory_format::any);
        assert(is_blocking_desc());
        const blocking_desc_t &blk = blocking_desc();
        const dims_t &optd = blk.offset_padding_to_data;

        size_t phys_offset = blk.offset_padding;
        for (int d = 0; d < ndims(); ++d) {
            const int block = blk.block_dims[d];

            const int p = pos[d] + (is_pos_padded ? 0 : optd[d]);
            const int pos_within_block = p % block;
            const int pos_block = p / block;

            phys_offset += pos_block * blk.strides[0][d];
            phys_offset += pos_within_block * blk.strides[1][d];
        }
        if (format() == gOIhw4i16o4i || format() == OIhw4i16o4i) {
            // TODO: Fix temporary workaround for formats with double blocking
            const bool with_groups = format() == gOIhw4i16o4i;
            const int oc_16 = pos[with_groups + 0] % 16;
            const int ic_4  = pos[with_groups + 1] % 4;
            phys_offset += 4 * oc_16 + ic_4 - (oc_16 + 16 * ic_4);
        }
        if (format() == gOIhw8i16o2i || format() == OIhw8i16o2i) {
            // TODO: Fix temporary workaround for formats with double blocking
            const bool with_groups = format() == gOIhw8i16o2i;
            const int oc_16 = pos[with_groups + 0] % 16;
            const int ic_2  = pos[with_groups + 1] % 2;
            phys_offset += -16 * ic_2 + oc_16 + ic_2;
        }
        if (format() == gOIdhw8i16o2i || format() == OIdhw8i16o2i) {
            // TODO: Fix temporary workaround for formats with double blocking
            const bool with_groups = format() == gOIdhw8i16o2i;
            const int oc_16 = pos[with_groups + 0] % 16;
            const int ic_2  = pos[with_groups + 1] % 2;
            phys_offset += -16 * ic_2 + oc_16 + ic_2;
        }
        if (format() == gOIhw8o16i2o || format() == OIhw8o16i2o) {
            // TODO: Fix temporary workaround for formats with double blocking
            const bool with_groups = format() == gOIhw8o16i2o;
            const int ic_16 = pos[with_groups + 1] % 16;
            const int oc_2  = pos[with_groups + 0] % 2;
            phys_offset += -16 * oc_2 + ic_16 + oc_2;
        }
        return phys_offset;
    }

    /** returns physical offset by logical one. logical offset is represented by
     * a scalar \param l_offset. if \param is_pos_padded is true, \param
     * l_offset represents logical offset in already padded area */
    inline size_t off_l(size_t l_offset, bool is_pos_padded = false) const {
        assert(is_blocking_desc());
        const dims_t &padding_dims = blocking_desc().padding_dims;
        dims_t pos;
        for (int rd = 0; rd < ndims(); ++rd) {
            const int d = ndims() - 1 - rd;
            const int cur_dim = is_pos_padded ? padding_dims[d] : dims()[d];
            pos[d] = l_offset % cur_dim;
            l_offset /= cur_dim;
        }
        return off_v(pos, is_pos_padded);
    }

    /** returns physical offset by logical one. logical offset is represented by
     * a tuple of indeces (\param xn, ..., \param x1, \param x0) */
    template<typename... Args> inline size_t off(Args... args) const {
        assert(sizeof...(args) == ndims());
        dims_t pos = { args... };
        return off_v(pos, false);
    }

    /** returns physical offset by logical one. logical offset is represented by
     * a tuple of indeces (\param xn, ..., \param x1, \param x0) in already
     * padded area */
    template<typename... Args> inline size_t off_padding(Args... args) const {
        assert(sizeof...(args) == ndims());
        dims_t pos = { args... };
        return off_v(pos, true);
    }

    /** returns physical offset by logical one. Logical offset is represented by
     * a tuple of block indeces (\param bn, ..., \param b1, \param b0). It is a
     * user responsibility to adjust the result to get offset within blocks */
    template<typename ...Args> inline size_t blk_off(Args... args) const {
        return _blk_off<sizeof...(args), Args...>(args...);
    }

    template<bool skip_first, typename T, typename ...Args>
    inline size_t blk_off(T xn, Args... args) const {
        return skip_first
            ? blk_off<Args...>(args...)
            : blk_off<T, Args...>(xn, args...);
    }

    /* static functions section */
    /* TODO: replace with non-static, once _md becomes non-const ref */

    static status_t compute_blocking(memory_desc_t &memory_desc);

private:
    /* TODO: put logical_offset in utils */
    template<typename T>
    inline size_t logical_offset(T x0) const { return size_t(x0); }

    template<typename T, typename... Args>
    inline size_t logical_offset(T xn, Args... args) const {
        const size_t n_args = sizeof...(args);
        return size_t(xn)*utils::array_product<n_args>(
                &dims()[ndims() - n_args]) + logical_offset(args...);
    }

    template<int ORIG_LEN, typename ...Void>
    inline size_t _blk_off() const {
        assert(is_blocking_desc());
        return blocking_desc().offset_padding;
    }

    template<int ORIG_LEN, typename T, typename ...Args>
    inline size_t _blk_off(T xc, Args ...args) const {
        assert(is_blocking_desc());
        constexpr int dc = ORIG_LEN - sizeof...(args) - 1;
        return size_t(xc) * blocking_desc().strides[0][dc]
            + _blk_off<ORIG_LEN, Args...>(args...);
    }
};

inline bool memory_desc_wrapper::is_dense(bool with_padding) const {
    if (utils::one_of(format(), memory_format::undef, memory_format::any))
        return false;
    return nelems(with_padding) * data_type_size() == size();
}

inline bool memory_desc_wrapper::operator==(const memory_desc_wrapper &rhs)
    const
{
    using namespace impl::types;
    return ndims() == rhs.ndims()
            && utils::array_cmp(dims(), rhs.dims(), ndims())
            && data_type() == rhs.data_type()
            && (is_blocking_desc()
                ? blocking_desc_is_equal(
                    blocking_desc(), rhs.blocking_desc(), ndims())
                : true)
            && (is_wino_desc()
                ? wino_desc_is_equal(wino_desc(), rhs.wino_desc()) : true);
}

inline bool memory_desc_wrapper::similar_to(const memory_desc_wrapper &rhs,
        bool with_padding, bool with_data_type, int dim_start) const {
    using namespace impl::types;
    using namespace utils;
    if (utils::one_of(format(), memory_format::undef, memory_format::any))
        return false;
    if (is_wino_desc() || rhs.is_wino_desc())
        return false;

    const int ds = dim_start;
    const auto &blk = blocking_desc();
    const auto &r_blk = rhs.blocking_desc();

    return ndims() == rhs.ndims()
        && dim_start <= ndims() /* guard */
        && array_cmp(dims() + ds, rhs.dims() + ds, ndims() - ds)
        && format_normalize(format()) == format_normalize(rhs.format())
        && implication(with_data_type, data_type() == rhs.data_type())
        && array_cmp(blk.block_dims + ds, r_blk.block_dims + ds, ndims() - ds)
        && array_cmp(blk.strides[0] + ds, r_blk.strides[0] + ds, ndims() - ds)
        && array_cmp(blk.strides[1] + ds, r_blk.strides[1] + ds, ndims() - ds)
        && implication(with_padding,
                array_cmp(blk.padding_dims + ds, r_blk.padding_dims + ds,
                    ndims() - ds)
                && array_cmp(blk.offset_padding_to_data + ds,
                    r_blk.offset_padding_to_data + ds, ndims() - ds));
}

inline bool memory_desc_wrapper::consistent_with(
        const memory_desc_wrapper &rhs) const {
    if (ndims() == rhs.ndims()) {
        for (int d = 0; d < ndims(); ++d) {
            if (dims()[d] != rhs.dims()[d]) return false;
        }
        return true;
    } else {
        /* TODO: revise.
         * is the following possible?
         * [1, a, b] <--reorder--> [a, b]
         * [a, 1, b] <--reorder--> [a, b]
         * not, at least for now */
        return false;
    }
}

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

