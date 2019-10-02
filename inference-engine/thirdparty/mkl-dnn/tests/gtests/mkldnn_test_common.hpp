/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
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

#ifndef MKLDNN_TEST_COMMON_HPP
#define MKLDNN_TEST_COMMON_HPP

#include <limits>
#include <numeric>
#include <vector>
#include <cmath>
#include <stdint.h>
#include<iostream>

#include "gtest/gtest.h"

#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#define collapse(x)
#endif

#include "mkldnn.hpp"

#include "src/common/mkldnn_thread.hpp"

#define SKIP_IF(cond, msg)                                      \
    do {                                                        \
        if (cond) {                                             \
            std::cout << "[ SKIPPED ] " << (msg) << std::endl; \
            return;                                             \
        }                                                       \
    } while (0)

template <typename data_t> struct data_traits { };
template <> struct data_traits<float> {
    static const auto data_type = mkldnn::memory::data_type::f32;
};
template <> struct data_traits<uint8_t> {
    static const auto data_type = mkldnn::memory::data_type::u8;
};
template <> struct data_traits<int8_t> {
    static const auto data_type = mkldnn::memory::data_type::s8;
};
template <> struct data_traits<int16_t> {
    static const auto data_type = mkldnn::memory::data_type::s16;
};
template <> struct data_traits<int32_t> {
    static const auto data_type = mkldnn::memory::data_type::s32;
};
template <> struct data_traits<mkldnn_bfloat16_t> {
    static const auto data_type = mkldnn::memory::data_type::bf16;
};

template <mkldnn::memory::data_type> struct prec_traits {};
template <> struct prec_traits<mkldnn::memory::data_type::f32> { typedef float type; };
template <> struct prec_traits<mkldnn::memory::data_type::s32> { typedef int32_t type; };
template <> struct prec_traits<mkldnn::memory::data_type::s16> { typedef int16_t type; };
template <> struct prec_traits<mkldnn::memory::data_type::s8> { typedef int8_t type; };
template <> struct prec_traits<mkldnn::memory::data_type::u8> { typedef uint8_t type; };
template <> struct prec_traits<mkldnn::memory::data_type::bf16> { typedef mkldnn_bfloat16_t type; };


template <typename T> inline void assert_eq(T a, T b);
template <> inline void assert_eq<float>(float a, float b) {
    ASSERT_FLOAT_EQ(a, b);
}

template <typename data_t> inline data_t out_round(float x,
        mkldnn_round_mode_t rmode = mkldnn_round_nearest)
{ return (data_t)(rmode == mkldnn_round_down ? floorf(x) : nearbyintf(x)); }
template <> inline float out_round<float>(float x, mkldnn_round_mode_t rmode)
{ (void)rmode; return x; }

template <typename data_t, typename out_t>
out_t saturate(const out_t &x) {
    out_t v = x;
    if (v <= std::numeric_limits<data_t>::min())
        v = std::numeric_limits<data_t>::min();
    if (v > std::numeric_limits<data_t>::max())
        v = std::numeric_limits<data_t>::max();
    return v;
}

inline int right_padding(int i, int o, int k, int p, int s, int d = 0) {
    return (o - 1) * s + (k - 1) * (d + 1) - (p + i - 1);
}

template <typename data_t> struct acc_t { typedef data_t type; };
template<> struct acc_t<int8_t> { typedef int type; };
template<> struct acc_t<uint8_t> { typedef int type; };

inline size_t map_index(const mkldnn::memory::desc &md, size_t index,
    bool with_padding = true) {
    using fmt = mkldnn::memory::format;

    const fmt fwd_weights_g_qvnni = fmt::gOIhw8i16o2i;
    const fmt fwd_weights_qvnni = fmt::OIhw8i16o2i;
    const fmt bwd_weights_g_qvnni = fmt::gOIhw8o16i2o;
    const fmt bwd_weights_qvnni = fmt::OIhw8o16i2o;

    const fmt fwd_weights_g_vnni = fmt::gOIhw4i16o4i;
    const fmt fwd_weights_vnni = fmt::OIhw4i16o4i;

    const bool with_groups = (md.data.format == fwd_weights_g_qvnni)
                          || (md.data.format == bwd_weights_g_qvnni)
                          || (md.data.format == fwd_weights_g_vnni);

    const bool qvnni = (md.data.format == fwd_weights_g_qvnni)
                    || (md.data.format == bwd_weights_g_qvnni)
                    || (md.data.format == fwd_weights_qvnni)
                    || (md.data.format == bwd_weights_qvnni);

    const bool vnni = (md.data.format == fwd_weights_g_vnni)
                   || (md.data.format == fwd_weights_vnni);

    const bool fwd_wei = (md.data.format == fwd_weights_g_qvnni)
                      || (md.data.format == fwd_weights_qvnni)
                      || (md.data.format == fwd_weights_g_vnni)
                      || (md.data.format == fwd_weights_vnni);

    const bool bwd_wei = (md.data.format == bwd_weights_g_qvnni)
                      || (md.data.format == bwd_weights_qvnni);

    const int ndims = md.data.ndims;
    const ptrdiff_t *dims = md.data.dims;
    const ptrdiff_t *pdims = md.data.layout_desc.blocking.padding_dims;
    const ptrdiff_t *optd = md.data.layout_desc.blocking.offset_padding_to_data;

    auto *strides_block = md.data.layout_desc.blocking.strides[0];
    auto *strides_within_block = md.data.layout_desc.blocking.strides[1];

    size_t ph_index = 0;
    size_t oc_lb = 0, ic_sb = 0,
        oc_sb = 0, ic_lb = 0;

    for (int rd = 0; rd < ndims; ++rd) {
        int d = ndims - rd - 1;

        EXPECT_LE(dims[d], pdims[d]);

        int cur_dim = with_padding ? pdims[d] : dims[d];
        EXPECT_GT(cur_dim, 0);
        int cur_block = md.data.layout_desc.blocking.block_dims[d];

        size_t pos_d = /*static_cast<ssize_t>*/(index % cur_dim);
        EXPECT_GE(optd[d], 0);
        size_t cur_pos = optd[d] + pos_d;

        size_t cur_pos_block = cur_pos / cur_block;
        size_t cur_pos_within_block = cur_pos % cur_block;

        if (d == (with_groups + 0)) {
            if (qvnni) { oc_lb = pos_d % 16;  oc_sb = pos_d % 2; }
            else  if (vnni) { oc_lb = pos_d % 16; }
        }
        if (d == (with_groups + 1)) {
            if (qvnni) { ic_sb = pos_d % 2; ic_lb = pos_d % 16; }
            else if (vnni) { ic_sb = pos_d % 4; }
        }
        ph_index += cur_pos_block*strides_block[d];
        ph_index += cur_pos_within_block*strides_within_block[d];

        index /= cur_dim;
    }
    int scale = (vnni) ? 3 : 1;
    if (fwd_wei) {
        //ph_index += -16 * ic_2 + oc_16 + ic_2;
        ph_index += scale * oc_lb + ic_sb;
        EXPECT_GE(ph_index, 16 * ic_sb);
        ph_index -= 16 * ic_sb;
    } else
        if (bwd_wei) {
            //ph_index += -16 * oc_2 + ic_16 + oc_2;
            ph_index += ic_lb + oc_sb;
            EXPECT_GE(ph_index, 16 * oc_sb);
            ph_index -= 16 * oc_sb;
        }
    ph_index += md.data.layout_desc.blocking.offset_padding;

    return ph_index;
}

#define MAX_NDIMS 12
// check_zero_tail - check on zero or set to zero padded memory
template <typename data_t>
void check_zero_tail(int set_zero_flag, mkldnn::memory &src) {

    data_t *src_data = (data_t *)src.get_data_handle();

    const mkldnn::memory::desc src_d = src.get_primitive_desc().desc();
    const int ndims = src_d.data.ndims;
    const ptrdiff_t *dims = src_d.data.dims;
    const ptrdiff_t *pdims = src_d.data.layout_desc.blocking.padding_dims;

    size_t idx[MAX_NDIMS] = {}, str[MAX_NDIMS] = {};
    size_t nelems = 1;
    int tail_flag = 0;
    for (int i = 0; i < ndims; ++i) {
        if (dims[ndims-i-1] != pdims[ndims-i-1]) tail_flag = 1;
        nelems *= pdims[ndims-i-1];
        idx[i] = 0;
        str[i] = (i==0) ? 1 : str[i-1] * pdims[ndims-i];
    }
    if (tail_flag == 0) return;

    for (size_t i = 0; i < nelems; ++i) {
        size_t off = 0;
        bool flag = 0;
        for (int j = 0; j < ndims; ++j) {
            off += idx[j] * str[j];
            if (idx[j] >= (size_t)dims[ndims-j-1]) flag = 1;
        }
        if (flag == 1) {
            size_t blk_off = map_index(src_d,off);
            if (set_zero_flag) {
                src_data[blk_off] = 0.0;
            } else {
                EXPECT_EQ(src_data[blk_off], 0.0) << " blk_off = " << blk_off
                << "off = " << off;
            }
        }
        /*Update idx*/
        for (int j = 0; j < ndims; ++j) {
            idx[j] ++;
            if (idx[j] < (size_t)pdims[ndims-j-1]) break;
            idx[j] = 0;
        }
    }
}

inline mkldnn::memory::desc create_md(mkldnn::memory::dims dims,
        mkldnn::memory::data_type data_type, mkldnn::memory::format fmt) {
    using f = mkldnn::memory::format;
    size_t ndims = 0;

    switch (fmt) {
    case f::x:
        ndims = 1; break;
    case f::nc:
    case f::oi:
    case f::io:
        ndims = 2; break;
    case f::nchw:
    case f::nhwc:
    case f::chwn:
    case f::nChw8c:
    case f::nChw16c:
    case f::oihw:
    case f::hwio:
    case f::iohw:
    case f::oIhw8i:
    case f::oIhw16i:
    case f::OIhw8i8o:
    case f::OIhw16i16o:
    case f::OIhw8i16o2i:
    case f::OIhw8o16i2o:
    case f::OIhw4i16o4i:
    case f::OIhw8o8i:
    case f::OIhw16o16i:
    case f::IOhw16o16i:
    case f::Ohwi8o:
    case f::Ohwi16o:
    case f::OhIw8o4i:
    case f::OIhw4i16o4i_s8s8:
    case f::OhIw8o4i_s8s8:
    case f::OhIw8o32i:
    case f::OhIw16o32i:
    case f::hwio_s8s8:
        ndims = 4; break;
    case f::ncdhw:
    case f::ndhwc:
    case f::nCdhw8c:
    case f::nCdhw16c:
    case f::dhwio:
    case f::oidhw:
    case f::goihw:
    case f::hwigo:
    case f::giohw:
    case f::oIdhw8i:
    case f::oIdhw16i:
    case f::OIdhw8i8o:
    case f::OIdhw16i16o:
    case f::OIdhw8o8i:
    case f::OIdhw16o16i:
    case f::gOhwi8o:
    case f::Goihw8g:
    case f::Goihw16g:
    case f::gOhwi16o:
    case f::gOIhw8i8o:
    case f::gOIhw16i16o:
    case f::gOIhw8i16o2i:
    case f::gOIhw8o16i2o:
    case f::gOIhw4i16o4i:
    case f::gOIhw8o8i:
    case f::gOIhw16o16i:
    case f::gIOhw16o16i:
    case f::gOhIw8o4i:
    case f::Goihw8g_s8s8:
    case f::Goihw16g_s8s8:
    case f::dhwio_s8s8:
    case f::OdhIw8o4i:
    case f::OdhIw8o4i_s8s8:
    case f::gOhIw8o4i_s8s8:
        ndims = 5; break;
    case f::gOIdhw8i8o:
    case f::gOIdhw16i16o:
    case f::gOIdhw8o8i:
    case f::gOIdhw16o16i:
    case f::gOdhwi16o:
    case f::goidhw:
    case f::dhwigo:
    case f::dhwigo_s8s8:
    case f::gOdhIw8o4i:
    case f::gOdhIw8o4i_s8s8:
    case f::Goidhw8g:
    case f::Goidhw8g_s8s8:
    case f::Goidhw16g:
    case f::Goidhw16g_s8s8:
        ndims = 6; break;
    case f::format_undef:
        ndims = 0; break;
    case f::any:
        return mkldnn::memory::desc(dims, data_type, fmt);
    default: EXPECT_TRUE(false) << "test does not support format: " << int(fmt);
    }

    EXPECT_EQ(dims.size(), ndims) << "dims and format are inconsistent";

    return mkldnn::memory::desc(dims, data_type, fmt);
}

union float_raw {
  float f;
  unsigned short i[2];
};

static void truncate_ps_to_bf16(mkldnn_bfloat16_t *out_bf16, const float *in_f32,
                    size_t length = 1) {
    for (size_t i = 0; i < length; i++) {
        union float_raw t = {0};
        t.f = in_f32[i];
        out_bf16[i] = t.i[1];
    }
}

static void cvt_bf16_to_ps(float *out_f32, const mkldnn_bfloat16_t *in_bf16,
                    size_t length = 1) {
    for (size_t i = 0; i < length; i++) {
        union float_raw t = {0};
        t.i[1] = in_bf16[i];
        t.i[0] = 0;
        out_f32[i] = t.f;
    }
}

template <typename data_t>
static inline data_t set_value(size_t index, data_t mean, data_t deviation,
        double sparsity)
{
    if (data_traits<data_t>::data_type == mkldnn::memory::data_type::f32) {
        const size_t group_size = (size_t)(1. / sparsity);
        const size_t group = index / group_size;
        const size_t in_group = index % group_size;
        const bool fill = in_group == ((group % 1637) % group_size);
        return fill ? static_cast<data_t>(mean + deviation * sinf(float(index % 37)))
            : data_t{0};
    } else if (data_traits<data_t>::data_type == mkldnn::memory::data_type::bf16) {
        const size_t group_size = (size_t)(1. / sparsity);
        const size_t group = index / group_size;
        const size_t in_group = index % group_size;
        const bool fill = in_group == ((group % 1637) % group_size);
        float val_f32 = fill ? mean + deviation * sinf(float(index % 37))
            : data_t{0};
        mkldnn_bfloat16_t val_bf16;
        truncate_ps_to_bf16(&val_bf16, &val_f32);
        return val_bf16;
    } else if (data_traits<data_t>::data_type == mkldnn::memory::data_type::s32
        || data_traits<data_t>::data_type == mkldnn::memory::data_type::s16
        || data_traits<data_t>::data_type == mkldnn::memory::data_type::s8) {
        return data_t(index * 13 % 21 - 10);
    } else if (data_traits<data_t>::data_type == mkldnn::memory::data_type::u8) {
        return data_t(index * 13 % 17);
    } else {
        return data_t(0);
    }
}

template <typename data_t>
static void fill_data(const size_t size, data_t *data, data_t mean,
        data_t deviation, double sparsity = 1.)
{
    mkldnn::impl::parallel_nd((ptrdiff_t)size, [&](ptrdiff_t n) {
            data[n] = set_value<data_t>(n, mean, deviation, sparsity);
    });
}

template <typename data_t>
static void fill_data(const size_t size, data_t *data, double sparsity = 1.,
        bool init_negs = false)
{
    mkldnn::impl::parallel_nd((ptrdiff_t)size, [&](ptrdiff_t n) {
        data[n] = set_value<data_t>(n, data_t(1), data_t(2e-1), sparsity);

        if (init_negs && n%4 == 0U)
            data[n] = static_cast<data_t>(-data[n]); // weird for unsigned types!
    });
}

int div_up(const int a, const int b) {
    return (a + b - 1) / b;
}

template <typename data_t>
inline void fill_data(const size_t size, data_t *data, const int init_range)
{
    for (size_t i = 0; i < size; i++) {
        float sign = (i % 3 == 0U) ? -1.0 : 1.0;
        data[i] = sign * ((i * 1637) % init_range);
    }

}

template <>
inline void fill_data<mkldnn_bfloat16_t>(const size_t size, mkldnn_bfloat16_t *data,
        const int init_range)
{
    union float_raw t = {0};
    for (size_t i = 0; i < size; i++) {
        float sign = (i % 3 == 0U) ? -1.0 : 1.0;
        t.f = sign * (float)((i * 1637) % init_range);
        data[i] = t.i[1];
    }

}

static inline void fill_data_bf16(const size_t size,
        mkldnn::memory& memory_bf16,
        mkldnn::memory& memory_f32,
        float mean = 1.0f, float deviation = 2.0e-1f, double sparsity = 1.) {
   fill_data<float>(size, (float *)memory_f32.get_data_handle(),
                    mean, deviation, sparsity);
   check_zero_tail<float>(1, memory_f32);
   truncate_ps_to_bf16((mkldnn_bfloat16_t *)memory_bf16.get_data_handle(),
       (float *)memory_f32.get_data_handle(), size);

   /* Adjust f32 data to be exactly representable in bf16
    * in order to get computed results which are bitwise accurate
    * when output data type is f32 */
   cvt_bf16_to_ps((float *)memory_f32.get_data_handle(),
           (mkldnn_bfloat16_t *)memory_bf16.get_data_handle(), size);
}

template <typename data_t>
static void compare_data(mkldnn::memory& ref, mkldnn::memory& dst,
        data_t threshold = (data_t)1e-4, bool isBinary = false)
{
    using data_type = mkldnn::memory::data_type;

    ASSERT_TRUE(data_traits<data_t>::data_type == data_type::f32 ||
                data_traits<data_t>::data_type == data_type::s32 ||
                data_traits<data_t>::data_type == data_type::u8);

    /* Note: size_t incompatible with MSVC++ */
    auto ref_desc = ref.get_primitive_desc().desc();
    auto dst_desc = dst.get_primitive_desc().desc();

    ASSERT_TRUE(ref_desc.data.ndims == dst_desc.data.ndims);

    auto ndims = ref_desc.data.ndims;

    for (auto d = 0; d < ndims; ++d) {
        ASSERT_TRUE(ref_desc.data.dims[d] == dst_desc.data.dims[d]);
    }

    auto dims = ref_desc.data.dims;

    ptrdiff_t num = 1;
    for (auto d = 0; d < ndims; ++d) {
        if (isBinary && d == 1) {
            num *= div_up(dims[d], 8);
        } else {
            num *= dims[d];
        }
    }

    data_t *ref_data = (data_t *)ref.get_data_handle();
    data_t *dst_data = (data_t *)dst.get_data_handle();

    mkldnn::impl::parallel_nd(num, [&](ptrdiff_t i) {
        int divider = isBinary ? 8 : 1;

        data_t ref = ref_data[map_index(ref_desc, i) / divider];
        data_t got = dst_data[map_index(dst_desc, i) / divider];

        if (data_traits<data_t>::data_type == data_type::f32) {
            data_t diff = got - ref;
            data_t e = (std::abs(ref) > threshold) ? diff / ref : diff;
            EXPECT_NEAR(e, (data_t) 0.0, threshold)
                                << "Index: " << i << " Total: " << num;
        } else {
            EXPECT_EQ(ref, got) << "Index: " << i << " Total: " << num;
        }
    });
}

inline const char *query_impl_info(const_mkldnn_primitive_desc_t pd) {
    const char *str;
    mkldnn_primitive_desc_query(pd, mkldnn_query_impl_info_str, 0, &str);
    return str;
};

mkldnn_status_t get_conv_impl_status(const_mkldnn_primitive_desc_t pd, const char *match_str){
    const char* conv_str = query_impl_info(pd);

    if( strstr(conv_str, match_str) != NULL)
        return mkldnn_status_t::mkldnn_success;
    return mkldnn_status_t::mkldnn_unimplemented;
};

struct test_convolution_sizes_t {
    test_convolution_sizes_t(
        int mb,
        int ng,
        int ic, int ih, int iw,
        int oc, int oh, int ow,
        int kh, int kw,
        int padh, int padw,
        int strh, int strw,
        int dilh=0, int dilw=0
    ) :
        mb(mb),
        ng(ng),
        ic(ic), ih(ih), iw(iw),
        oc(oc), oh(oh), ow(ow),
        kh(kh), kw(kw),
        padh(padh), padw(padw),
        strh(strh), strw(strw),
        dilh(dilh), dilw(dilw) {}
    int mb;
    int ng;
    int ic, ih, iw;
    int oc, oh, ow;
    int kh, kw;
    int padh, padw;
    int strh, strw;
    int dilh, dilw;
};

struct test_convolution_sizes_t_3d {
    test_convolution_sizes_t_3d(
        int mb,
        int ng,
        int ic, int id, int ih, int iw,
        int oc, int od, int oh, int ow,
        int kd, int kh, int kw,
        int padd, int padh, int padw,
        int strd, int strh, int strw,
        int dild=0, int dilh=0, int dilw=0
    ) :
        mb(mb),
        ng(ng),
        ic(ic), id(id), ih(ih), iw(iw),
        oc(oc), od(od), oh(oh), ow(ow),
        kd(kd), kh(kh), kw(kw),
        padd(padd), padh(padh), padw(padw),
        strd(strd), strh(strh), strw(strw),
        dild(dild), dilh(dilh), dilw(dilw) {}
    int mb;
    int ng;
    int ic, id, ih, iw;
    int oc, od, oh, ow;
    int kd, kh, kw;
    int padd, padh, padw;
    int strd, strh, strw;
    int dild, dilh, dilw;
};

struct test_convolution_attr_t {
    struct scale_t {
        enum policy_t { NONE = 0, COMMON };

        bool is_def() const { return policy != NONE; }

        scale_t (float s, policy_t p = NONE) :
            scale(s) { policy = p; }

        policy_t policy;
        float scale;
    };

    void mkldnn_attr_recreate() {
        mkl_attr = mkldnn::primitive_attr();
        mkl_attr.set_int_output_round_mode(rmode);
        if (oscale.is_def()) {
            const int count = 1;
            const int mask = 0;
            std::vector<float> s(count, oscale.scale);
            mkl_attr.set_output_scales(mask, s);
        }
    }

    test_convolution_attr_t(mkldnn::round_mode rm, float s,
        scale_t::policy_t p = scale_t::policy_t::NONE) :
            rmode(rm), oscale(s, p), mkl_attr() {}

    test_convolution_attr_t() :
        rmode(mkldnn::round_mode::round_nearest),
        oscale(1.0), mkl_attr() {}

    mkldnn::round_mode rmode;
    scale_t oscale;
    mkldnn::primitive_attr mkl_attr;
};

struct test_convolution_formats_t {
    mkldnn::memory::format src_format;
    mkldnn::memory::format weights_format;
    mkldnn::memory::format bias_format;
    mkldnn::memory::format dst_format;
};

struct test_convolution_params_t {
    const mkldnn::engine::kind engine_kind;
    mkldnn::algorithm aalgorithm;
    test_convolution_formats_t formats;
    test_convolution_attr_t attr;
    test_convolution_sizes_t sizes;
    bool expect_to_fail;
    mkldnn_status_t expected_status;
};

struct test_convolution_params_t_3d {
    const mkldnn::engine::kind engine_kind;
    mkldnn::algorithm aalgorithm;
    test_convolution_formats_t formats;
    test_convolution_attr_t attr;
    test_convolution_sizes_t_3d sizes;
    bool expect_to_fail;
    mkldnn_status_t expected_status;
};

struct test_convolution_eltwise_params_t {
    const mkldnn::algorithm alg;
    const mkldnn::engine::kind engine_kind;
    mkldnn::algorithm aalgorithm;
    const float eltwise_alpha;
    const float eltwise_beta;
    test_convolution_formats_t formats;
    test_convolution_attr_t attr;
    test_convolution_sizes_t sizes;
    bool expect_to_fail;
    mkldnn_status_t expected_status;
};

struct test_convolution_depthwise_params_t {
    const mkldnn::algorithm alg;
    const mkldnn::engine::kind engine_kind;
    mkldnn::algorithm aalgorithm;
    test_convolution_formats_t formats;
    test_convolution_attr_t attr;
    test_convolution_sizes_t sizes;
    bool expect_to_fail;
    mkldnn_status_t expected_status;
};

struct test_convolution_dw_conv_sizes_t {
    test_convolution_dw_conv_sizes_t(
            int mb, int ic, int ih, int iw,
            int conv1_oc,
            int conv1_kh, int conv1_kw,
            int conv1_padh, int conv1_padw,
            int conv1_strh, int conv1_strw,
            int conv2_oc,
            int conv2_kh, int conv2_kw,
            int conv2_padh, int conv2_padw,
            int conv2_strh, int conv2_strw
    ) :
            mb(mb), ic(ic), ih(ih), iw(iw),
            conv1_oc(conv1_oc),
            conv1_kh(conv1_kh), conv1_kw(conv1_kw),
            conv1_padh(conv1_padh), conv1_padw(conv1_padw),
            conv1_strh(conv1_strh), conv1_strw(conv1_strw),
            conv2_oc(conv2_oc),
            conv2_kh(conv2_kh), conv2_kw(conv2_kw),
            conv2_padh(conv2_padh), conv2_padw(conv2_padw),
            conv2_strh(conv2_strh), conv2_strw(conv2_strw) {}
    int mb, ic, ih, iw;
    int conv1_oc;
    int conv1_kh,   conv1_kw;
    int conv1_padh, conv1_padw;
    int conv1_strh, conv1_strw;
    int conv2_oc;
    int conv2_kh,   conv2_kw;
    int conv2_padh, conv2_padw;
    int conv2_strh, conv2_strw;
};

struct test_convolution_dw_conv_formats_t {
    mkldnn::memory::format src_format;
    mkldnn::memory::format conv1_weights_format;
    mkldnn::memory::format conv1_bias_format;
    mkldnn::memory::format conv2_weights_format;
    mkldnn::memory::format conv2_bias_format;
    mkldnn::memory::format dst_format;
};

struct test_convolution_dw_conv_params_t {
    const mkldnn::engine::kind engine_kind;
    mkldnn::algorithm aalgorithm;
    test_convolution_dw_conv_formats_t formats;
    test_convolution_dw_conv_sizes_t sizes;
};

struct test_roi_pool_desc_t {
    struct {
        int mb, c;
        int h, w;
    } data;

    struct {
        int mb, c;
        int h, w;
    } roi;

    int pooled_h, pooled_w;
    double spatial_scale;
};

struct roi_pool_test_params {
    mkldnn::prop_kind aprop_kind;
    mkldnn::algorithm algorithm_kind;
    const mkldnn::engine::kind engine_kind;
    mkldnn::memory::format data_format;
    mkldnn::memory::format roi_format;
    mkldnn::memory::format dst_format;
    test_roi_pool_desc_t test_pd;
};

struct test_binary_convolution_params_t {
    const mkldnn::engine::kind engine_kind;
    mkldnn::algorithm aalgorithm;
    float pad_value;
    mkldnn::algorithm eltwise_algorithm;
    const float eltwise_alpha;
    const float eltwise_beta;
    mkldnn::algorithm depthwise_algorithm;
    bool with_sum;
    mkldnn::algorithm binarization_algorithm;
    test_convolution_formats_t formats;
    test_convolution_sizes_t sizes;
};

struct test_binary_convolution_dw_conv_params_t {
    const mkldnn::engine::kind engine_kind;
    mkldnn::algorithm aalgorithm;
    mkldnn::algorithm eltwise_algorithm;
    const float eltwise_alpha;
    const float eltwise_beta;
    mkldnn::algorithm depthwise_algorithm;
    bool with_sum;
    mkldnn::algorithm binarization_algorithm;
    test_convolution_dw_conv_formats_t formats;
    test_convolution_dw_conv_sizes_t sizes;
};

struct test_deformable_convolution_formats_t {
    mkldnn::memory::format src_format;
    mkldnn::memory::format offsets_format;
    mkldnn::memory::format weights_format;
    mkldnn::memory::format bias_format;
    mkldnn::memory::format dst_format;
};

struct test_deformable_convolution_sizes_t {
    test_deformable_convolution_sizes_t(
        int mb,
        int ng,
        int dg,
        int ic, int ih, int iw,
        int oc, int oh, int ow,
        int kh, int kw,
        int padh, int padw,
        int strh, int strw,
        int dilh=0, int dilw=0
    ) :
        mb(mb),
        ng(ng),
        dg(dg),
        ic(ic), ih(ih), iw(iw),
        oc(oc), oh(oh), ow(ow),
        kh(kh), kw(kw),
        padh(padh), padw(padw),
        strh(strh), strw(strw),
        dilh(dilh), dilw(dilw) {}
    int mb;
    int ng;
    int dg;
    int ic, ih, iw;
    int oc, oh, ow;
    int kh, kw;
    int padh, padw;
    int strh, strw;
    int dilh, dilw;
};

struct test_deformable_convolution_params_t {
    const mkldnn::engine::kind engine_kind;
    mkldnn::algorithm aalgorithm;
    test_deformable_convolution_formats_t formats;
    test_deformable_convolution_sizes_t sizes;
};

std::ostream &operator<<(std::ostream &stream,
                         const roi_pool_test_params &tp)
{
    return stream << "(" << "input_data:" << " mb = " << tp.test_pd.data.mb  << ", c = " << tp.test_pd.data.c
                  << ", h = " << tp.test_pd.data.h   << ", w = " << tp.test_pd.data.w
                  << ", rois_num: " << tp.test_pd.roi.mb
                  << ", pooled_h: " << tp.test_pd.pooled_h
                  << ", pooled_w: " << tp.test_pd.pooled_w
                  << ", spatial_scale: " << tp.test_pd.spatial_scale
                  << ")";
}

template<typename F> bool catch_expected_failures(const F &f,
        bool expect_to_fail, mkldnn_status_t expected_status, bool ignore_unimplemented = true)
{
    try {
        f();
    } catch (const mkldnn::error &e) {
        // Rethrow the exception if it is not expected or the error status did
        // not match.
        if (!(expect_to_fail) || e.status != (expected_status)) {
            // Ignore unimplemented
            if ( ignore_unimplemented && (e.status == mkldnn_unimplemented))
                return true;
            else
                throw e;
        }
        // Return normally if the failure is expected
        if (expect_to_fail)
            return true;
    }

    // Throw an exception if the failure is expected but did not happen
    if (expect_to_fail)
        throw std::exception();

    return false;
}

#define TEST_MALLOC_OFFSET 8
char *test_malloc(size_t size) {
    void *ptr;
    const size_t align = 64;
    const size_t padded_size = TEST_MALLOC_OFFSET + size;
#ifdef _WIN32
    ptr = _aligned_malloc(padded_size, align);
    int rc = ((ptr) ? 0 : errno);
#else
    int rc = ::posix_memalign(&ptr, align, padded_size);
#endif /* _WIN32 */
    return rc == 0 ? (char*)ptr + TEST_MALLOC_OFFSET: 0;
}

void test_free(char *ptr) {
    char *base_ptr = ptr - TEST_MALLOC_OFFSET;
#ifdef _WIN32
    _aligned_free(base_ptr);
#else
    return ::free(base_ptr);
#endif /* _WIN32 */
}
#undef TEST_MALLOC_OFFSET

class test_memory {
public:
    test_memory(const mkldnn::memory::desc &d, const mkldnn::engine &e) {
        auto pd = mkldnn::memory::primitive_desc(d, e);
        pd_size_ = pd.get_size();
        data_.reset(test_malloc(pd_size_), test_free);
        mem_.reset(new mkldnn::memory(pd, data_.get()));
    }
    size_t get_size() const { return pd_size_; }
    mkldnn::memory &get() { return *mem_; }

private:
    std::shared_ptr<mkldnn::memory> mem_;
    std::shared_ptr<char> data_;
    size_t pd_size_;
};

#endif
