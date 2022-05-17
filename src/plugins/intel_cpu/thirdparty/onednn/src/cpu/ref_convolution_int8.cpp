/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/ref_io_helper.hpp"
#include "cpu/simple_q10n.hpp"

#include "cpu/ref_convolution_int8.hpp"
#include "cpu/ref_convolution_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

status_t ref_convolution_int8_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DST, status);
    CHECK(status);

    DEFINE_ZERO_POINTS_BUFFER(src_zero_point, DNNL_ARG_SRC);
    DEFINE_ZERO_POINTS_BUFFER(dst_zero_point, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const bool with_groups = pd()->with_groups();

    const auto G = pd()->G();
    const auto MB = pd()->MB();
    const auto OD = pd()->OD();
    const auto OH = pd()->OH();
    const auto OW = pd()->OW();
    const auto ID = pd()->ID();
    const auto IH = pd()->IH();
    const auto IW = pd()->IW();

    const auto OC = pd()->OC() / G;
    const auto IC = pd()->IC() / G;
    const auto KD = pd()->KD();
    const auto KH = pd()->KH();
    const auto KW = pd()->KW();

    const auto KSD = pd()->KSD();
    const auto KSH = pd()->KSH();
    const auto KSW = pd()->KSW();

    const auto KDD = pd()->KDD() + 1;
    const auto KDH = pd()->KDH() + 1;
    const auto KDW = pd()->KDW() + 1;

    const auto padFront = pd()->padFront();
    const auto padT = pd()->padT();
    const auto padL = pd()->padL();

    const auto ndims = pd()->desc()->src_desc.ndims;

    auto maybe_oscale = [=](float &d, dim_t g, dim_t oc) {
        // scale_idx_mult = 1 for per_oc scales and 0, otherwise
        const int scale_idx_mult
                = pd()->attr()->output_scales_.mask_ == (1 << 1);
        const float *scales = pd()->attr()->output_scales_.scales_;
        d *= scales[(g * OC + oc) * scale_idx_mult];
    };

    // zp_idx_mult = 1 for per_dim1 zero points and 0, otherwise
    const int src_zp_idx_mult
            = !pd()->attr()->zero_points_.common(DNNL_ARG_SRC);
    const int dst_zp_idx_mult
            = !pd()->attr()->zero_points_.common(DNNL_ARG_DST);

    auto ker = [=](dim_t g, dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow) {
        int d = 0;
        for_(dim_t ic = 0; ic < IC; ++ic)
        for_(dim_t kd = 0; kd < KD; ++kd)
        for_(dim_t kh = 0; kh < KH; ++kh)
        for (dim_t kw = 0; kw < KW; ++kw) {
            const dim_t id = od * KSD - padFront + kd * KDD;
            const dim_t ih = oh * KSH - padT + kh * KDH;
            const dim_t iw = ow * KSW - padL + kw * KDW;

            if (id < 0 || id >= ID) continue;
            if (ih < 0 || ih >= IH) continue;
            if (iw < 0 || iw >= IW) continue;

            const auto src_off = ref_conv_utils::get_data_off(
                    src_d, ndims, mb, g * IC + ic, id, ih, iw);
            const auto wei_off = ref_conv_utils::get_weights_off(
                    weights_d, with_groups, ndims, g, oc, ic, kd, kh, kw);

            const int s = io::load_int_value(src_d.data_type(), src, src_off);
            const int src_zp = src_zero_point
                    ? io::load_int_value(data_type::s32, src_zero_point,
                            src_zp_idx_mult * (g * IC + ic))
                    : 0;
            const int w = io::load_int_value(
                    weights_d.data_type(), weights, wei_off);
            d += (s - src_zp) * w;
        }
        return d;
    };

    // help compiler optimize the code constants for plain layouts kernel
    const dims_t &src_str = src_d.blocking_desc().strides;
    const dim_t src_ic_stride = src_str[1];
    const dim_t src_id_stride = (ndims == 5) ? src_str[2] : 0;
    const dim_t src_ih_stride = (ndims >= 4) ? src_str[ndims - 2] : 0;
    const dim_t src_iw_stride = (ndims >= 3) ? src_str[ndims - 1] : 0;
    const dims_t &weights_str = weights_d.blocking_desc().strides;
    const int gr_shift = with_groups ? 1 : 0;
    const dim_t weights_ic_stride = weights_str[1 + gr_shift];
    const dim_t weights_kd_stride
            = (ndims == 5) ? weights_str[2 + gr_shift] : 0;
    const dim_t weights_kh_stride
            = (ndims >= 4) ? weights_str[ndims - 2 + gr_shift] : 0;
    const dim_t weights_kw_stride
            = (ndims >= 3) ? weights_str[ndims - 1 + gr_shift] : 0;

    auto ker_plain = [=](dim_t g, dim_t mb, dim_t oc, dim_t od, dim_t oh,
                             dim_t ow) {
        assert(3 <= ndims && ndims <= 5);
        int d = 0;

        const dim_t src_loc_off = ref_conv_utils::get_data_off(
                src_d, ndims, mb, g * IC, 0, 0, 0);
        const dim_t weights_loc_off = ref_conv_utils::get_weights_off(
                weights_d, with_groups, ndims, g, oc, 0, 0, 0, 0);

        const void *__restrict src_loc = src;
        const void *__restrict weights_loc = weights;

        if (IC > KW) {
            for_(dim_t kd = 0; kd < KD; ++kd)
            for_(dim_t kh = 0; kh < KH; ++kh)
            for (dim_t kw = 0; kw < KW; ++kw) {
                const dim_t id = od * KSD - padFront + kd * KDD;
                const dim_t ih = oh * KSH - padT + kh * KDH;
                const dim_t iw = ow * KSW - padL + kw * KDW;
                if (id < 0 || id >= ID || ih < 0 || ih >= IH || iw < 0
                        || iw >= IW)
                    continue;

                for (dim_t ic = 0; ic < IC; ++ic) {
                    const dim_t src_off = ic + id * src_id_stride
                            + ih * src_ih_stride + iw * src_iw_stride;
                    const dim_t weights_off = ic * weights_ic_stride
                            + kd * weights_kd_stride + kh * weights_kh_stride
                            + kw;
                    const int s = io::load_int_value(
                            src_d.data_type(), src_loc, src_off + src_loc_off);
                    const int src_zp = src_zero_point
                            ? io::load_int_value(data_type::s32, src_zero_point,
                                    src_zp_idx_mult * (g * IC + ic))
                            : 0;
                    const int w = io::load_int_value(weights_d.data_type(),
                            weights_loc, weights_off + weights_loc_off);
                    d += (s - src_zp) * w;
                }
            }
        } else {
            for_(dim_t ic = 0; ic < IC; ++ic)
            for_(dim_t kd = 0; kd < KD; ++kd)
            for_(dim_t kh = 0; kh < KH; ++kh)
            for (dim_t kw = 0; kw < KW; ++kw) {
                const dim_t id = od * KSD - padFront + kd * KDD;
                const dim_t ih = oh * KSH - padT + kh * KDH;
                const dim_t iw = ow * KSW - padL + kw * KDW;
                if (id < 0 || id >= ID || ih < 0 || ih >= IH || iw < 0
                        || iw >= IW)
                    continue;

                const dim_t src_off = ic + id * src_id_stride
                        + ih * src_ih_stride + iw * src_iw_stride;
                const dim_t weights_off = ic * weights_ic_stride
                        + kd * weights_kd_stride + kh * weights_kh_stride + kw;
                const int s = io::load_int_value(
                        src_d.data_type(), src_loc, src_off + src_loc_off);
                const int src_zp = src_zero_point
                        ? io::load_int_value(data_type::s32, src_zero_point,
                                src_zp_idx_mult * (g * IC + ic))
                        : 0;
                const int w = io::load_int_value(weights_d.data_type(),
                        weights_loc, weights_off + weights_loc_off);
                d += (s - src_zp) * w;
            }
        }
        return d;
    };

    const auto sum_dt = pd()->attr()->post_ops_.get_sum_dt(dst_d.data_type());

    parallel_nd(G, MB, OC, OD, OH, OW,
            [&](dim_t g, dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow) {
                int acc = 0;
                if (src_d.is_plain() && weights_d.is_plain()
                        && src_ic_stride == 1 && weights_kw_stride == 1)
                    acc += ker_plain(g, mb, oc, od, oh, ow);
                else
                    acc += ker(g, mb, oc, od, oh, ow);

                float d = static_cast<float>(acc);
                if (bias) {
                    const auto bias_off = bias_d.off(g * OC + oc);
                    const float b = io::load_float_value(
                            bias_d.data_type(), bias, bias_off);
                    d += b;
                }

                dim_t dst_off = ref_conv_utils::get_data_off(
                        dst_d, ndims, mb, g * OC + oc, od, oh, ow);

                dim_t dst_l_off = (mb * OC * G + g * OC + oc) * OD * OH * OW
                        + od * OH * OW + oh * OW + ow;

                maybe_oscale(d, g, oc);

                ref_post_ops_t::args_t args;
                args.dst_val = io::load_float_value(sum_dt, dst, dst_off);
                args.ctx = &ctx;
                args.l_offset = dst_l_off;
                args.dst_md = pd()->dst_md();
                ref_post_ops->execute(d, args);

                if (dst_zero_point) {
                    const int dst_zp = io::load_int_value(data_type::s32,
                            dst_zero_point, dst_zp_idx_mult * (g * OC + oc));
                    d += dst_zp;
                }
                io::store_float_value(dst_d.data_type(), d, dst, dst_off);
            });

    return status::success;
}

status_t ref_convolution_int8_bwd_data_t::execute_backward_data(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DIFF_SRC, status);
    CHECK(status);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const bool with_groups = pd()->with_groups();

    const auto G = pd()->G();
    const auto MB = pd()->MB();
    const auto OD = pd()->OD();
    const auto OH = pd()->OH();
    const auto OW = pd()->OW();
    const auto ID = pd()->ID();
    const auto IH = pd()->IH();
    const auto IW = pd()->IW();

    const auto OC = pd()->OC() / G;
    const auto IC = pd()->IC() / G;
    const auto KD = pd()->KD();
    const auto KH = pd()->KH();
    const auto KW = pd()->KW();

    const auto KSD = pd()->KSD();
    const auto KSH = pd()->KSH();
    const auto KSW = pd()->KSW();

    const auto KDD = pd()->KDD() + 1;
    const auto KDH = pd()->KDH() + 1;
    const auto KDW = pd()->KDW() + 1;

    const auto padFront = pd()->padFront();
    const auto padT = pd()->padT();
    const auto padL = pd()->padL();

    const auto ndims = pd()->desc()->diff_src_desc.ndims;

    auto maybe_oscale = [=](float &ds, dim_t g, dim_t ic) {
        /* scale_idx_mult = 1 for per_oc scales and 0, otherwise */
        const int scale_idx_mult
                = pd()->attr()->output_scales_.mask_ == (1 << 1);
        const float *scales = pd()->attr()->output_scales_.scales_;
        ds *= scales[(g * IC + ic) * scale_idx_mult];
    };

    auto ker = [=](dim_t g, dim_t mb, dim_t ic, dim_t id, dim_t ih, dim_t iw) {
        int ds = 0;
        for_(dim_t oc = 0; oc < OC; ++oc)
        for_(dim_t kd = 0; kd < KD; ++kd)
        for_(dim_t kh = 0; kh < KH; ++kh)
        for (dim_t kw = 0; kw < KW; ++kw) {
            if (iw + padL < kw * KDW || ih + padT < kh * KDH
                    || id + padFront < kd * KDD)
                continue;
            dim_t ow = iw - kw * KDW + padL;
            dim_t oh = ih - kh * KDH + padT;
            dim_t od = id - kd * KDD + padFront;
            if (ow % KSW != 0 || oh % KSH != 0 || od % KSD != 0) continue;

            ow /= KSW;
            oh /= KSH;
            od /= KSD;

            if (od < OD && oh < OH && ow < OW) {
                const auto diff_dst_off = ref_conv_utils::get_data_off(
                        diff_dst_d, ndims, mb, g * OC + oc, od, oh, ow);
                const auto weights_off = ref_conv_utils::get_weights_off(
                        weights_d, with_groups, ndims, g, oc, ic, kd, kh, kw);
                const int dd = io::load_int_value(
                        diff_dst_d.data_type(), diff_dst, diff_dst_off);
                const int w = io::load_int_value(
                        weights_d.data_type(), weights, weights_off);
                ds += dd * w;
            }
        }
        return ds;
    };

    // help compiler optimize the code constants for plain layouts kernel
    const dims_t &diff_dst_str = diff_dst_d.blocking_desc().strides;
    const dim_t diff_dst_oc_stride = diff_dst_str[1];
    const dim_t diff_dst_ow_stride = diff_dst_str[ndims - 1];
    const dim_t diff_dst_oh_stride = (ndims >= 4) ? diff_dst_str[ndims - 2] : 0;
    const dim_t diff_dst_od_stride = (ndims >= 5) ? diff_dst_str[ndims - 3] : 0;

    const dims_t &weights_str = weights_d.blocking_desc().strides;
    const int gr_shift = with_groups ? 1 : 0;
    const dim_t weights_oc_stride = weights_str[0 + gr_shift];
    const dim_t weights_kw_stride = weights_str[ndims - 1 + gr_shift];
    const dim_t weights_kh_stride
            = (ndims >= 4) ? weights_str[ndims - 2 + gr_shift] : 0;
    const dim_t weights_kd_stride
            = (ndims >= 5) ? weights_str[ndims - 3 + gr_shift] : 0;

    auto ker_plain = [=](dim_t g, dim_t mb, dim_t ic, dim_t id, dim_t ih,
                             dim_t iw) {
        assert(3 <= ndims && ndims <= 5);
        int ds = 0;
        const dim_t diff_dst_loc_off = ref_conv_utils::get_data_off(
                diff_dst_d, ndims, mb, g * OC, 0, 0, 0);
        const dim_t weights_loc_off = ref_conv_utils::get_weights_off(
                weights_d, with_groups, ndims, g, 0, ic, 0, 0, 0);

        const void *__restrict diff_dst_loc = diff_dst;
        const void *__restrict weights_loc = weights;

        if (OC > KW) {
            for_(dim_t kd = 0; kd < KD; ++kd)
            for_(dim_t kh = 0; kh < KH; ++kh)
            for (dim_t kw = 0; kw < KW; ++kw) {
                dim_t ow = iw - kw * KDW + padL;
                dim_t oh = ih - kh * KDH + padT;
                dim_t od = id - kd * KDD + padFront;
                if (ow < 0 || oh < 0 || od < 0 || ow % KSW != 0 || oh % KSH != 0
                        || od % KSD != 0)
                    continue;
                ow /= KSW;
                oh /= KSH;
                od /= KSD;
                if (od >= OD || oh >= OH || ow >= OW) continue;
                for (dim_t oc = 0; oc < OC; ++oc) {
                    const dim_t diff_dst_off = oc + od * diff_dst_od_stride
                            + oh * diff_dst_oh_stride + ow * diff_dst_ow_stride;
                    const dim_t weights_off = oc * weights_oc_stride
                            + kd * weights_kd_stride + kh * weights_kh_stride
                            + kw;
                    const int dd = io::load_int_value(diff_dst_d.data_type(),
                            diff_dst_loc, diff_dst_off + diff_dst_loc_off);
                    const int w = io::load_int_value(weights_d.data_type(),
                            weights_loc, weights_off + weights_loc_off);
                    ds += dd * w;
                }
            }
        } else {
            for_(dim_t oc = 0; oc < OC; ++oc)
            for_(dim_t kd = 0; kd < KD; ++kd)
            for (dim_t kh = 0; kh < KH; ++kh) {
                // Note: placing these 2 params outside the `kw-loop` because
                // of a compiler-generated bug. Declaring 'od' as volatile
                // fixes a recurring seg-fault.
                const volatile dim_t od_ = id - kd * KDD + padFront;
                const dim_t weights_off_ = oc * weights_oc_stride
                        + kd * weights_kd_stride + kh * weights_kh_stride;
                for (dim_t kw = 0; kw < KW; ++kw) {
                    dim_t ow = iw - kw * KDW + padL;
                    dim_t oh = ih - kh * KDH + padT;
                    dim_t od = od_;
                    if (ow < 0 || oh < 0 || od < 0 || ow % KSW != 0
                            || oh % KSH != 0 || od % KSD != 0)
                        continue;
                    ow /= KSW;
                    oh /= KSH;
                    od /= KSD;
                    if (od >= OD || oh >= OH || ow >= OW) continue;
                    const dim_t diff_dst_off = oc + od * diff_dst_od_stride
                            + oh * diff_dst_oh_stride + ow * diff_dst_ow_stride;
                    const dim_t weights_off = weights_off_ + kw;
                    const int dd = io::load_int_value(diff_dst_d.data_type(),
                            diff_dst_loc, diff_dst_off + diff_dst_loc_off);
                    const int w = io::load_int_value(weights_d.data_type(),
                            weights_loc, weights_off + weights_loc_off);
                    ds += dd * w;
                }
            }
        }
        return ds;
    };

    parallel_nd(G, MB, IC, ID, IH, IW,
            [&](dim_t g, dim_t mb, dim_t ic, dim_t id, dim_t ih, dim_t iw) {
                int acc = 0;
                if (diff_dst_d.is_plain() && weights_d.is_plain()
                        && diff_dst_oc_stride == 1 && weights_kw_stride == 1)
                    acc += ker_plain(g, mb, ic, id, ih, iw);
                else
                    acc += ker(g, mb, ic, id, ih, iw);

                float ds = static_cast<float>(acc);
                maybe_oscale(ds, g, ic);

                const auto diff_src_off = ref_conv_utils::get_data_off(
                        diff_src_d, ndims, mb, g * IC + ic, id, ih, iw);
                io::store_float_value(
                        diff_src_d.data_type(), ds, diff_src, diff_src_off);
            });

    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
