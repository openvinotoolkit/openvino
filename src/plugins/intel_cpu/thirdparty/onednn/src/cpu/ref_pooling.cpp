/*******************************************************************************
* Copyright 2016-2021 Intel Corporation
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
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"

#include "cpu/simple_q10n.hpp"

#include "cpu/ref_pooling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

static inline dim_t get_offset(const memory_desc_wrapper &mdw, dim_t n, dim_t c,
        dim_t d, dim_t h, dim_t w) {
    switch (mdw.ndims()) {
        case 3: return mdw.off(n, c, w);
        case 4: return mdw.off(n, c, h, w);
        case 5: return mdw.off(n, c, d, h, w);
        default: assert(!"Invalid tensor dimension in pooling");
    }
    return 0;
}

using namespace nstl;

template <data_type_t src_type, data_type_t dst_type, data_type_t acc_type>
status_t ref_pooling_fwd_t<src_type, dst_type, acc_type>::execute_forward(
        const exec_ctx_t &ctx) const {

    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);
    auto ws = CTX_OUT_MEM(unsigned char *, DNNL_ARG_WORKSPACE);

    auto MB = CTX_IN_BATCH(DNNL_ARG_SRC);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper ws_d(pd()->workspace_md());

    const data_type_t ws_dt = ws ? ws_d.data_type() : data_type::undef;
    if (ws) assert(ws_dt == data_type::u8 || ws_dt == data_type::s32);

    const auto alg = pd()->desc()->alg_kind;
    const dim_t OC = pd()->OC();
    const dim_t OD = pd()->OD();
    const dim_t OH = pd()->OH();
    const dim_t OW = pd()->OW();
    const dim_t ID = pd()->ID();
    const dim_t IH = pd()->IH();
    const dim_t IW = pd()->IW();
    const dim_t KD = pd()->KD();
    const dim_t KH = pd()->KH();
    const dim_t KW = pd()->KW();
    const dim_t SD = pd()->KSD();
    const dim_t SH = pd()->KSH();
    const dim_t SW = pd()->KSW();
    const dim_t padF = pd()->padFront();
    const dim_t padT = pd()->padT();
    const dim_t padL = pd()->padL();
    const dim_t DD = pd()->KDD();
    const dim_t DH = pd()->KDH();
    const dim_t DW = pd()->KDW();
    const dim_t padB = pd()->padB();
    const dim_t padR = pd()->padR();
    const dim_t padBack = pd()->padBack();

    auto set_ws = [=](dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow,
                          dim_t value) {
        if (ws) {
            const auto off = get_offset(ws_d, mb, oc, od, oh, ow);
            if (ws_dt == data_type::u8) {
                assert(0 <= value
                        && value <= numeric_limits<typename prec_traits<
                                        data_type::u8>::type>::max());
                ws[off] = value;
            } else
                reinterpret_cast<int *>(ws)[off] = value;
        }
    };

    auto ker_max = [=](float &d, dim_t mb, dim_t oc, dim_t od, dim_t oh,
                           dim_t ow) {
        bool is_initialized = false;
        set_ws(mb, oc, od, oh, ow, 0);
        for (dim_t kd = 0; kd < KD; ++kd) {
            const dim_t id = od * SD - padF + kd * (DD + 1);
            if (id < 0 || id >= ID) continue;
            for (dim_t kh = 0; kh < KH; ++kh) {
                const dim_t ih = oh * SH - padT + kh * (DH + 1);
                if (ih < 0 || ih >= IH) continue;
                for (dim_t kw = 0; kw < KW; ++kw) {
                    const dim_t iw = ow * SW - padL + kw * (DW + 1);
                    if (iw < 0 || iw >= IW) continue;

                    const auto off = get_offset(src_d, mb, oc, id, ih, iw);
                    auto s = src[off];
                    if (!is_initialized) {
                        d = s;
                        set_ws(mb, oc, od, oh, ow, kd * KH * KW + kh*KW + kw);
                        is_initialized = true;
                    } else {
                        if (s > d) {
                            d = s;
                            set_ws(mb, oc, od, oh, ow, kd * KH * KW + kh * KW + kw);
                        }
                    }
                }
            }
        }
    };

    auto ker_avg = [=](float &d, dim_t mb, dim_t oc, dim_t od, dim_t oh,
                           dim_t ow) {
        for (dim_t kd = 0; kd < KD; ++kd) {
            const dim_t id = od * SD - padF + kd * (DD + 1);
            if (id < 0 || id >= ID) continue;
            for (dim_t kh = 0; kh < KH; ++kh) {
                const dim_t ih = oh * SH - padT + kh * (DH + 1);
                if (ih < 0 || ih >= IH) continue;
                for (dim_t kw = 0; kw < KW; ++kw) {
                    const dim_t iw = ow * SW - padL + kw * (DW + 1);
                    if (iw < 0 || iw >= IW) continue;

                    const auto off = get_offset(src_d, mb, oc, id, ih, iw);
                    d += src[off];
                }
            }
        }

        auto id_start = od*SD - padF;
        auto ih_start = oh*SH - padT;
        auto iw_start = ow*SW - padL;
        auto id_end = nstl::min(od*SD - padF + KD, ID + padBack);
        auto ih_end = nstl::min(oh*SH - padT + KH, IH + padB);
        auto iw_end = nstl::min(ow*SW - padL + KW, IW + padR);

        // case alg == pooling_avg_include_padding
        auto num_summands = (ih_end - ih_start)*(iw_end - iw_start)*(id_end - id_start);

        id_start = nstl::max(id_start, dim_t(0));
        ih_start = nstl::max(ih_start, dim_t(0));
        iw_start = nstl::max(iw_start, dim_t(0));
        id_end = nstl::min(id_end, ID);
        ih_end = nstl::min(ih_end, IH);
        iw_end = nstl::min(iw_end, IW);

        if (alg == alg_kind::pooling_avg_exclude_padding) {
            auto id_start_excluded
                    = id_start < 0 ? (0 - id_start - 1) / (DD + 1) + 1 : 0;
            auto ih_start_excluded
                    = ih_start < 0 ? (0 - ih_start - 1) / (DH + 1) + 1 : 0;
            auto iw_start_excluded
                    = iw_start < 0 ? (0 - iw_start - 1) / (DW + 1) + 1 : 0;
            auto id_end_excluded
                    = id_end > ID ? (id_end - ID - 1) / (DD + 1) + 1 : 0;
            auto ih_end_excluded
                    = ih_end > IH ? (ih_end - IH - 1) / (DH + 1) + 1 : 0;
            auto iw_end_excluded
                    = iw_end > IW ? (iw_end - IW - 1) / (DW + 1) + 1 : 0;

            num_summands = (KD - id_start_excluded - id_end_excluded)
                           * (KH - ih_start_excluded - ih_end_excluded)
                           * (KW - iw_start_excluded - iw_end_excluded);
        }
        if (num_summands == 0) return;

        d /= num_summands;

        const auto &p = pd()->attr()->post_ops_;
        for (int i = 0; i < p.len(); i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_quantization()) {
                auto quant = post_op.quantization;
                auto quantization_base = CTX_IN_MEM(const float *, (DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1));
                const auto crop_low_data =  quantization_base + quant.offset[quant.crop_low];
                const auto crop_high_data =  quantization_base + quant.offset[quant.crop_high];
                const auto inp_scale_data = quantization_base + quant.offset[quant.inp_scale];
                const auto inp_shift_data = quantization_base + quant.offset[quant.inp_shift];
                const auto output_scale_data = quantization_base + quant.offset[quant.output_scale];
                const auto output_shift_data = quantization_base + quant.offset[quant.output_shift];

                float cl = crop_low_data[!quant.per_channel[quant.crop_low] ? 0 : oc];
                float ch = crop_high_data[!quant.per_channel[quant.crop_high] ? 0 : oc];
                float isc = inp_scale_data[!quant.per_channel[quant.inp_scale] ? 0 : oc];
                float ish = inp_shift_data[!quant.per_channel[quant.inp_shift] ? 0 : oc];
                float osc = output_scale_data[!quant.per_channel[quant.output_scale] ? 0 : oc];
                float osh = output_shift_data[!quant.per_channel[quant.output_shift] ? 0 : oc];

                d = nstl::min(ch, nstl::max(cl, d));
                d = d * isc + ish;
                d = roundf(d);
                d = d * osc + osh;
            }
        }
    };

    const bool is_max_pool = alg == alg_kind::pooling_max;

    using ker_t
            = std::function<void(float &, dim_t, dim_t, dim_t, dim_t, dim_t)>;
    ker_t kernel = is_max_pool ? (ker_t)ker_max : (ker_t)ker_avg;

    parallel_nd(MB, OC, OD, OH, OW,
            [&](dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow) {
                auto data_p_off = get_offset(dst_d, mb, oc, od, oh, ow);
                auto data_l_off
                        = (((mb * OC + oc) * OD + od) * OH + oh) * OW + ow;
                float res = 0.f;
                kernel(res, mb, oc, od, oh, ow);

                ref_post_ops_t::args_t args;
                args.ctx = &ctx;
                args.l_offset = data_l_off;
                args.dst_md = pd()->dst_md();
                ref_post_ops->execute(res, args, oc);

                dst[data_p_off] = cpu::saturate_and_round<dst_data_t>(res);
            });

    return status::success;
}

template <data_type_t data_type>
status_t ref_pooling_bwd_t<data_type>::execute_backward(
        const exec_ctx_t &ctx) const {

    status_t status = status::success;

    const auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    const auto ws = CTX_IN_MEM(const unsigned char *, DNNL_ARG_WORKSPACE);
    auto diff_src = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_DIFF_SRC, status);
    CHECK(status);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper ws_d(pd()->workspace_md());

    const auto alg = pd()->desc()->alg_kind;
    const dim_t MB = pd()->MB();
    const dim_t OC = pd()->OC();
    const dim_t OD = pd()->OD();
    const dim_t OH = pd()->OH();
    const dim_t OW = pd()->OW();
    const dim_t ID = pd()->ID();
    const dim_t IH = pd()->IH();
    const dim_t IW = pd()->IW();
    const dim_t KD = pd()->KD();
    const dim_t KH = pd()->KH();
    const dim_t KW = pd()->KW();
    const dim_t SD = pd()->KSD();
    const dim_t SH = pd()->KSH();
    const dim_t SW = pd()->KSW();
    const dim_t padF = pd()->padFront();
    const dim_t padT = pd()->padT();
    const dim_t padL = pd()->padL();
    const dim_t DD = pd()->KDD();
    const dim_t DH = pd()->KDH();
    const dim_t DW = pd()->KDW();

    auto ker_zero = [=](dim_t mb, dim_t oc) {
        for_(dim_t id = 0; id < ID; ++id)
        for_(dim_t ih = 0; ih < IH; ++ih)
        for (dim_t iw = 0; iw < IW; ++iw) {
            const auto off = get_offset(diff_src_d, mb, oc, id, ih, iw);
            diff_src[off] = data_type_t(0);
        }
    };

    auto ker_max = [=](dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow) {
        const auto ws_off = get_offset(ws_d, mb, oc, od, oh, ow);
        const int index = ws_d.data_type() == data_type::u8
                ? (int)ws[ws_off]
                : ((int *)ws)[ws_off];
        const dim_t kd = (index / KW) / KH;
        const dim_t kh = (index / KW) % KH;
        const dim_t kw = index % KW;
        const dim_t id = od * SD - padF + kd * (DD + 1);
        const dim_t ih = oh * SH - padT + kh * (DH + 1);
        const dim_t iw = ow * SW - padL + kw * (DW + 1);

        // If padding area could fit the kernel,
        // then input displacement would be out of bounds.
        // No need to back propagate there as padding is
        // virtual in pooling_max case.
        if (id < 0 || id >= ID) return;
        if (ih < 0 || ih >= IH) return;
        if (iw < 0 || iw >= IW) return;

        const auto d_src_off = get_offset(diff_src_d, mb, oc, id, ih, iw);
        const auto d_dst_off = get_offset(diff_dst_d, mb, oc, od, oh, ow);
        diff_src[d_src_off] += diff_dst[d_dst_off];
    };

    auto ker_avg = [=](dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow) {
        int num_summands;
        if (alg == alg_kind::pooling_avg_include_padding)
            num_summands = KW * KH * KD;
        else {
            auto id_start = od * SD - padF;
            auto ih_start = oh * SH - padT;
            auto iw_start = ow * SW - padL;
            auto id_end = od * SD - padF + (KD - 1) * DD + KD;
            auto ih_end = oh * SH - padT + (KH - 1) * DH + KH;
            auto iw_end = ow * SW - padL + (KW - 1) * DW + KW;

            auto id_start_excluded
                    = id_start < 0 ? (0 - id_start - 1) / (DD + 1) + 1 : 0;
            auto ih_start_excluded
                    = ih_start < 0 ? (0 - ih_start - 1) / (DH + 1) + 1 : 0;
            auto iw_start_excluded
                    = iw_start < 0 ? (0 - iw_start - 1) / (DW + 1) + 1 : 0;
            auto id_end_excluded
                    = id_end > ID ? (id_end - ID - 1) / (DD + 1) + 1 : 0;
            auto ih_end_excluded
                    = ih_end > IH ? (ih_end - IH - 1) / (DH + 1) + 1 : 0;
            auto iw_end_excluded
                    = iw_end > IW ? (iw_end - IW - 1) / (DW + 1) + 1 : 0;

            num_summands = (KD - id_start_excluded - id_end_excluded)
                    * (KH - ih_start_excluded - ih_end_excluded)
                    * (KW - iw_start_excluded - iw_end_excluded);
        }
        for (dim_t kd = 0; kd < KD; ++kd) {
            const dim_t id = od * SD - padF + kd * (DD + 1);
            if (id < 0 || id >= ID) continue;
            for (dim_t kh = 0; kh < KH; ++kh) {
                const dim_t ih = oh * SH - padT + kh * (DH + 1);
                if (ih < 0 || ih >= IH) continue;
                for (dim_t kw = 0; kw < KW; ++kw) {
                    const dim_t iw = ow * SW - padL + kw * (DW + 1);
                    if (iw < 0 || iw >= IW) continue;

                    const auto d_src_off
                            = get_offset(diff_src_d, mb, oc, id, ih, iw);
                    const auto d_dst_off
                            = get_offset(diff_dst_d, mb, oc, od, oh, ow);
                    diff_src[d_src_off] += diff_dst[d_dst_off] / num_summands;
                }
            }
        }
    };

    dim_t ow_start
            = max(dim_t(0), utils::div_up(padL - ((KW - 1) * DW + KW) + 1, SW));
    dim_t ow_end = min(OW, 1 + (padL + IW - 1) / SW);

    dim_t oh_start
            = max(dim_t(0), utils::div_up(padT - ((KH - 1) * DH + KH) + 1, SH));
    dim_t oh_end = min(OH, 1 + (padT + IH - 1) / SH);

    dim_t od_start
            = max(dim_t(0), utils::div_up(padF - ((KD - 1) * DD + KD) + 1, SD));
    dim_t od_end = min(OD, 1 + (padF + ID - 1) / SD);

    using ker_t = std::function<void(dim_t, dim_t, dim_t, dim_t, dim_t)>;
    ker_t kernel
            = alg == alg_kind::pooling_max ? (ker_t)ker_max : (ker_t)ker_avg;

    parallel_nd(MB, OC, [&](dim_t mb, dim_t oc) {
        ker_zero(mb, oc);
        for_(dim_t od = od_start; od < od_end; ++od)
        for_(dim_t oh = oh_start; oh < oh_end; ++oh)
        for (dim_t ow = ow_start; ow < ow_end; ++ow) {
            kernel(mb, oc, od, oh, ow);
        }
    });

    return status::success;
}

template struct ref_pooling_fwd_t<data_type::f32, data_type::f32, data_type::f32>;
template struct ref_pooling_fwd_t<data_type::s32, data_type::s32, data_type::s32>;
template struct ref_pooling_fwd_t<data_type::bf16, data_type::bf16, data_type::f32>;
template struct ref_pooling_fwd_t<data_type::s8, data_type::s8, data_type::s32>;
template struct ref_pooling_fwd_t<data_type::u8, data_type::u8, data_type::s32>;
template struct ref_pooling_fwd_t<data_type::s8, data_type::f32, data_type::f32>;
template struct ref_pooling_fwd_t<data_type::u8, data_type::f32, data_type::f32>;

template struct ref_pooling_bwd_t<data_type::f32>;
template struct ref_pooling_bwd_t<data_type::bf16>;
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
