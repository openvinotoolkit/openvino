/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include <cassert>

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/simple_resampling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace format_tag;
using namespace resampling_utils;
using namespace std::placeholders;

using namespace resampling_utils;

namespace {

template <data_type_t src_type, data_type_t dst_type>
struct simple_resampling_kernel_t : public simple_resampling_base_t {
    simple_resampling_kernel_t(const resampling_pd_t *pd);

    using src_data_t = typename prec_traits<src_type>::type;
    using dst_data_t = typename prec_traits<dst_type>::type;

    status_t init() override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    using interpolate_fn_t = std::function<void(const src_data_t *,
            dst_data_t *, ref_post_ops_t::args_t &, dim_t, dim_t, dim_t)>;

    void fill_coeffs();
    void fill_weights();
    interpolate_fn_t create_nearest() const;
    interpolate_fn_t create_linear() const;
    interpolate_fn_t create_bilinear() const;
    interpolate_fn_t create_trilinear() const;

    // For fwd processing:
    const bool are_postops_set_;
    const ref_post_ops_t ref_post_ops_;
    std::vector<linear_coeffs_t> linear_coeffs_;

    // For bwd processing:
    std::vector<float> bwd_linear_weights_;
    std::vector<bwd_linear_coeffs_t> bwd_linear_coeffs_;

    interpolate_fn_t interpolate_fn_;
};

template <data_type_t src_type, data_type_t dst_type>
simple_resampling_kernel_t<src_type, dst_type>::simple_resampling_kernel_t(
        const resampling_pd_t *pd)
    : simple_resampling_base_t(pd)
    , are_postops_set_(!(pd_->attr()->post_ops_.entry_.empty()))
    , ref_post_ops_(pd_->attr()->post_ops_) {
    if (pd_->is_fwd()) {
        const memory_desc_wrapper src_d(pd_->src_md());
        inner_stride_ = src_d.blocking_desc().strides[pd_->ndims() - 1];
        nsp_outer_ = src_d.nelems(true)
                / (pd_->ID() * pd_->IH() * pd_->IW() * inner_stride_);
        stride_d_ = pd_->IH() * pd_->IW() * inner_stride_;
        stride_h_ = pd_->IW() * inner_stride_;
        stride_w_ = inner_stride_;
    } else {
        const memory_desc_wrapper diff_src_d(pd_->diff_src_md());
        inner_stride_ = diff_src_d.blocking_desc().strides[pd_->ndims() - 1];
        nsp_outer_ = diff_src_d.nelems(true)
                / (pd_->ID() * pd_->IH() * pd_->IW() * inner_stride_);
        stride_d_ = pd_->OH() * pd_->OW() * inner_stride_;
        stride_h_ = pd_->OW() * inner_stride_;
        stride_w_ = inner_stride_;
    }
}

template <data_type_t src_type, data_type_t dst_type>
status_t simple_resampling_kernel_t<src_type, dst_type>::init() {
    if (pd_->desc()->alg_kind == alg_kind::resampling_nearest)
        interpolate_fn_ = create_nearest();
    else {
        if (pd_->ndims() == 5)
            interpolate_fn_ = create_trilinear();
        else if (pd_->ndims() == 4)
            interpolate_fn_ = create_bilinear();
        else
            interpolate_fn_ = create_linear();

        fill_coeffs();
        if (!pd_->is_fwd()) fill_weights();
    }

    return status::success;
}

template <data_type_t src_type, data_type_t dst_type>
status_t simple_resampling_kernel_t<src_type, dst_type>::execute(
        const exec_ctx_t &ctx) const {
    const int OD = pd_->OD();
    const int OH = pd_->OH();
    const int OW = pd_->OW();
    const int ID = pd_->ID();
    const int IH = pd_->IH();
    const int IW = pd_->IW();

    if (pd_->is_fwd()) {
        const auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
        auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

        parallel_nd(nsp_outer_, OD, OH, [&](dim_t nsp0, dim_t od, dim_t oh) {
            ref_post_ops_t::args_t postops_args;
            postops_args.ctx = &ctx;
            postops_args.dst_md = pd_->dst_md();

            for (dim_t ow = 0; ow < OW; ow++) {
                const dim_t src_off = nsp0 * ID * IH * IW * inner_stride_;
                const dim_t dst_off
                        = (nsp0 * OD * OH * OW + od * OH * OW + oh * OW + ow)
                        * inner_stride_;

                postops_args.l_offset = dst_off;

                interpolate_fn_(
                        src + src_off, dst + dst_off, postops_args, od, oh, ow);
            }
        });
    } else {
        const auto diff_dst = CTX_IN_MEM(const src_data_t *, DNNL_ARG_DIFF_DST);
        auto diff_src = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DIFF_SRC);
        ref_post_ops_t::args_t empty_args;

        parallel_nd(nsp_outer_, ID, IH, IW,
                [&](dim_t nsp, dim_t id, dim_t ih, dim_t iw) {
                    const dim_t diff_dst_off
                            = nsp * OD * OH * OW * inner_stride_;
                    const dim_t diff_src_off
                            = (nsp * ID * IH * IW + id * IH * IW + ih * IW + iw)
                            * inner_stride_;
                    interpolate_fn_(diff_dst + diff_dst_off,
                            diff_src + diff_src_off, empty_args, id, ih, iw);
                });
    }

    return status::success;
}

template <data_type_t src_type, data_type_t dst_type>
void simple_resampling_kernel_t<src_type, dst_type>::fill_coeffs() {
    if (pd_->is_fwd()) {
        linear_coeffs_.reserve(pd_->OD() + pd_->OH() + pd_->OW());
        for (dim_t od = 0; od < pd_->OD(); od++)
            linear_coeffs_.emplace_back(
                    linear_coeffs_t(od, pd_->OD(), pd_->ID()));
        for (dim_t oh = 0; oh < pd_->OH(); oh++)
            linear_coeffs_.emplace_back(
                    linear_coeffs_t(oh, pd_->OH(), pd_->IH()));
        for (dim_t ow = 0; ow < pd_->OW(); ow++)
            linear_coeffs_.emplace_back(
                    linear_coeffs_t(ow, pd_->OW(), pd_->IW()));
    } else {
        bwd_linear_coeffs_.reserve(pd_->ID() + pd_->IH() + pd_->IW());
        for (dim_t id = 0; id < pd_->ID(); id++)
            bwd_linear_coeffs_.emplace_back(
                    bwd_linear_coeffs_t(id, pd_->OD(), pd_->ID()));
        for (dim_t ih = 0; ih < pd_->IH(); ih++)
            bwd_linear_coeffs_.emplace_back(
                    bwd_linear_coeffs_t(ih, pd_->OH(), pd_->IH()));
        for (dim_t iw = 0; iw < pd_->IW(); iw++)
            bwd_linear_coeffs_.emplace_back(
                    bwd_linear_coeffs_t(iw, pd_->OW(), pd_->IW()));
    }
}

template <data_type_t src_type, data_type_t dst_type>
void simple_resampling_kernel_t<src_type, dst_type>::fill_weights() {
    assert(!pd_->is_fwd() && "The function is used in bwd path only.");

    using namespace resampling_utils;
    bwd_linear_weights_.reserve(2 * (pd_->OD() + pd_->OH() + pd_->OW()));
    for (dim_t od = 0; od < pd_->OD(); od++) {
        bwd_linear_weights_.emplace_back(
                linear_weight(0, od, pd_->OD(), pd_->ID()));
        bwd_linear_weights_.emplace_back(
                linear_weight(1, od, pd_->OD(), pd_->ID()));
    }
    for (dim_t oh = 0; oh < pd_->OH(); oh++) {
        bwd_linear_weights_.emplace_back(
                linear_weight(0, oh, pd_->OH(), pd_->IH()));
        bwd_linear_weights_.emplace_back(
                linear_weight(1, oh, pd_->OH(), pd_->IH()));
    }
    for (dim_t ow = 0; ow < pd_->OW(); ow++) {
        bwd_linear_weights_.emplace_back(
                linear_weight(0, ow, pd_->OW(), pd_->IW()));
        bwd_linear_weights_.emplace_back(
                linear_weight(1, ow, pd_->OW(), pd_->IW()));
    }
}

template <data_type_t src_type, data_type_t dst_type>
typename simple_resampling_kernel_t<src_type, dst_type>::interpolate_fn_t
simple_resampling_kernel_t<src_type, dst_type>::create_nearest() const {
    if (pd_->is_fwd()) {
        return [&](const src_data_t *src, dst_data_t *dst,
                       ref_post_ops_t::args_t &po_args, dim_t od, dim_t oh,
                       dim_t ow) {
            const dim_t id = nearest_idx(od, pd_->OD(), pd_->ID());
            const dim_t ih = nearest_idx(oh, pd_->OH(), pd_->IH());
            const dim_t iw = nearest_idx(ow, pd_->OW(), pd_->IW());
            const dim_t offset
                    = id * stride_d_ + ih * stride_h_ + iw * stride_w_;

            PRAGMA_OMP_SIMD()
            for (dim_t innermost_el = 0; innermost_el < inner_stride_;
                    innermost_el++) {
                float res = static_cast<float>(src[offset + innermost_el]);

                if (are_postops_set_) {
                    po_args.dst_val = dst[innermost_el];
                    ref_post_ops_.execute(res, po_args);
                    po_args.l_offset++;
                }

                dst[innermost_el] = cpu::saturate_and_round<dst_data_t>(res);
            }
        };
    } else {
        return [&](const src_data_t *diff_dst, dst_data_t *diff_src,
                       ref_post_ops_t::args_t &po_args, dim_t id, dim_t ih,
                       dim_t iw) {
            auto ow_idx = [&](const float in_idx) -> dim_t {
                return ceil_idx((in_idx * pd_->OW() / pd_->IW()) - 0.5f);
            };
            auto oh_idx = [&](const float in_idx) -> dim_t {
                return ceil_idx((in_idx * pd_->OH() / pd_->IH()) - 0.5f);
            };
            auto od_idx = [&](const float in_idx) -> dim_t {
                return ceil_idx((in_idx * pd_->OD() / pd_->ID()) - 0.5f);
            };

            const dim_t ow_start = ow_idx(iw) * stride_w_;
            const dim_t oh_start = oh_idx(ih) * stride_h_;
            const dim_t od_start = od_idx(id) * stride_d_;
            const dim_t ow_end = ow_idx(iw + 1.f) * stride_w_;
            const dim_t oh_end = oh_idx(ih + 1.f) * stride_h_;
            const dim_t od_end = od_idx(id + 1.f) * stride_d_;

            PRAGMA_OMP_SIMD()
            for (dim_t innermost_el = 0; innermost_el < inner_stride_;
                    innermost_el++) {
                float sum = 0;
                for_(dim_t od = od_start; od < od_end; od += stride_d_)
                for_(dim_t oh = oh_start; oh < oh_end; oh += stride_h_)
                for (dim_t ow = ow_start; ow < ow_end; ow += stride_w_) {
                    sum += static_cast<float>(
                            diff_dst[od + oh + ow + innermost_el]);
                }
                diff_src[innermost_el]
                        = cpu::saturate_and_round<dst_data_t>(sum);
            }
        };
    }
}

template <data_type_t src_type, data_type_t dst_type>
typename simple_resampling_kernel_t<src_type, dst_type>::interpolate_fn_t
simple_resampling_kernel_t<src_type, dst_type>::create_linear() const {
    if (pd_->is_fwd()) {
        return [&](const src_data_t *src, dst_data_t *dst,
                       ref_post_ops_t::args_t &po_args, dim_t od, dim_t oh,
                       dim_t ow) {
            const linear_coeffs_t &iw
                    = linear_coeffs_[pd_->OD() + pd_->OH() + ow];

            PRAGMA_OMP_SIMD()
            for (dim_t innermost_el = 0; innermost_el < inner_stride_;
                    innermost_el++) {
                float res = 0;
                for (int k = 0; k < 2; k++)
                    res += static_cast<float>(
                                   src[iw.idx[k] * stride_w_ + innermost_el])
                            * iw.wei[k];

                if (are_postops_set_) {
                    po_args.dst_val = dst[innermost_el];
                    ref_post_ops_.execute(res, po_args);
                    po_args.l_offset++;
                }

                dst[innermost_el] = cpu::saturate_and_round<dst_data_t>(res);
            }
        };
    } else {
        return [&](const src_data_t *diff_dst, dst_data_t *diff_src,
                       ref_post_ops_t::args_t &po_args, dim_t id, dim_t ih,
                       dim_t iw) {
            const bwd_linear_coeffs_t &w
                    = bwd_linear_coeffs_[pd_->ID() + pd_->IH() + iw];

            PRAGMA_OMP_SIMD()
            for (dim_t innermost_el = 0; innermost_el < inner_stride_;
                    innermost_el++) {
                float sum = 0;
                for_(int k = 0; k < 2; k++)
                for (dim_t ow = w.start[k]; ow < w.end[k]; ow++) {
                    sum += static_cast<float>(
                                   diff_dst[ow * stride_w_ + innermost_el])
                            * bwd_linear_weights_[2
                                            * (pd_->OD() + pd_->OH() + ow)
                                    + k];
                }
                diff_src[innermost_el]
                        = cpu::saturate_and_round<dst_data_t>(sum);
            }
        };
    }
}

template <data_type_t src_type, data_type_t dst_type>
typename simple_resampling_kernel_t<src_type, dst_type>::interpolate_fn_t
simple_resampling_kernel_t<src_type, dst_type>::create_bilinear() const {
    if (pd_->is_fwd()) {
        return [&](const src_data_t *src, dst_data_t *dst,
                       ref_post_ops_t::args_t &po_args, dim_t od, dim_t oh,
                       dim_t ow) {
            const linear_coeffs_t &ih = linear_coeffs_[pd_->OD() + oh];
            const linear_coeffs_t &iw
                    = linear_coeffs_[pd_->OD() + pd_->OH() + ow];

            PRAGMA_OMP_SIMD()
            for (dim_t innermost_el = 0; innermost_el < inner_stride_;
                    innermost_el++) {
                float res = 0;
                for_(int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++)
                    res += static_cast<float>(src[ih.idx[j] * stride_h_
                                   + iw.idx[k] * stride_w_ + innermost_el])
                            * ih.wei[j] * iw.wei[k];

                if (are_postops_set_) {
                    po_args.dst_val = dst[innermost_el];
                    ref_post_ops_.execute(res, po_args);
                    po_args.l_offset++;
                }

                dst[innermost_el] = cpu::saturate_and_round<dst_data_t>(res);
            }
        };
    } else {
        return [&](const src_data_t *diff_dst, dst_data_t *diff_src,
                       ref_post_ops_t::args_t &po_args, dim_t id, dim_t ih,
                       dim_t iw) {
            const bwd_linear_coeffs_t &h = bwd_linear_coeffs_[pd_->ID() + ih];
            const bwd_linear_coeffs_t &w
                    = bwd_linear_coeffs_[pd_->ID() + pd_->IH() + iw];

            PRAGMA_OMP_SIMD()
            for (dim_t innermost_el = 0; innermost_el < inner_stride_;
                    innermost_el++) {
                float sum = 0;
                for_(int j = 0; j < 2; j++)
                for_(int k = 0; k < 2; k++)
                for_(dim_t oh = h.start[j]; oh < h.end[j]; oh++)
                for (dim_t ow = w.start[k]; ow < w.end[k]; ow++) {
                    sum += static_cast<float>(diff_dst[oh * stride_h_
                                   + ow * stride_w_ + innermost_el])
                            * bwd_linear_weights_[2 * (pd_->OD() + oh) + j]
                            * bwd_linear_weights_[2
                                            * (pd_->OD() + pd_->OH() + ow)
                                    + k];
                }
                diff_src[innermost_el]
                        = cpu::saturate_and_round<dst_data_t>(sum);
            }
        };
    }
}

template <data_type_t src_type, data_type_t dst_type>
typename simple_resampling_kernel_t<src_type, dst_type>::interpolate_fn_t
simple_resampling_kernel_t<src_type, dst_type>::create_trilinear() const {
    if (pd_->is_fwd()) {
        return [&](const src_data_t *src, dst_data_t *dst,
                       ref_post_ops_t::args_t &po_args, dim_t od, dim_t oh,
                       dim_t ow) {
            const linear_coeffs_t &id = linear_coeffs_[od];
            const linear_coeffs_t &ih = linear_coeffs_[pd_->OD() + oh];
            const linear_coeffs_t &iw
                    = linear_coeffs_[pd_->OD() + pd_->OH() + ow];

            PRAGMA_OMP_SIMD()
            for (dim_t innermost_el = 0; innermost_el < inner_stride_;
                    innermost_el++) {
                float res = 0;
                for_(int i = 0; i < 2; i++)
                for_(int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++)
                    res += static_cast<float>(src[id.idx[i] * stride_d_
                                   + ih.idx[j] * stride_h_
                                   + iw.idx[k] * stride_w_ + innermost_el])
                            * id.wei[i] * ih.wei[j] * iw.wei[k];

                if (are_postops_set_) {
                    po_args.dst_val = dst[innermost_el];
                    ref_post_ops_.execute(res, po_args);
                    po_args.l_offset++;
                }

                dst[innermost_el] = cpu::saturate_and_round<dst_data_t>(res);
            }
        };
    } else {
        return [&](const src_data_t *diff_dst, dst_data_t *diff_src,
                       ref_post_ops_t::args_t &po_args, dim_t id, dim_t ih,
                       dim_t iw) {
            const bwd_linear_coeffs_t &d = bwd_linear_coeffs_[id];
            const bwd_linear_coeffs_t &h = bwd_linear_coeffs_[pd_->ID() + ih];
            const bwd_linear_coeffs_t &w
                    = bwd_linear_coeffs_[pd_->ID() + pd_->IH() + iw];

            PRAGMA_OMP_SIMD()
            for (dim_t innermost_el = 0; innermost_el < inner_stride_;
                    innermost_el++) {
                float sum = 0;
                for_(int i = 0; i < 2; i++)
                for_(int j = 0; j < 2; j++)
                for_(int k = 0; k < 2; k++)
                for_(dim_t od = d.start[i]; od < d.end[i]; od++)
                for_(dim_t oh = h.start[j]; oh < h.end[j]; oh++)
                for (dim_t ow = w.start[k]; ow < w.end[k]; ow++) {
                    sum += static_cast<float>(
                                   diff_dst[od * stride_d_ + oh * stride_h_
                                           + ow * stride_w_ + innermost_el])
                            * bwd_linear_weights_[2 * od + i]
                            * bwd_linear_weights_[2 * (pd_->OD() + oh) + j]
                            * bwd_linear_weights_[2
                                            * (pd_->OD() + pd_->OH() + ow)
                                    + k];
                }
                diff_src[innermost_el]
                        = cpu::saturate_and_round<dst_data_t>(sum);
            }
        };
    }
}

template struct simple_resampling_kernel_t<data_type::f32, data_type::f32>;
template struct simple_resampling_kernel_t<data_type::f32, data_type::bf16>;
template struct simple_resampling_kernel_t<data_type::f32, data_type::s32>;
template struct simple_resampling_kernel_t<data_type::f32, data_type::s8>;
template struct simple_resampling_kernel_t<data_type::f32, data_type::u8>;
template struct simple_resampling_kernel_t<data_type::bf16, data_type::f32>;
template struct simple_resampling_kernel_t<data_type::bf16, data_type::bf16>;
template struct simple_resampling_kernel_t<data_type::bf16, data_type::s32>;
template struct simple_resampling_kernel_t<data_type::bf16, data_type::s8>;
template struct simple_resampling_kernel_t<data_type::bf16, data_type::u8>;
template struct simple_resampling_kernel_t<data_type::s32, data_type::f32>;
template struct simple_resampling_kernel_t<data_type::s32, data_type::bf16>;
template struct simple_resampling_kernel_t<data_type::s32, data_type::s32>;
template struct simple_resampling_kernel_t<data_type::s32, data_type::s8>;
template struct simple_resampling_kernel_t<data_type::s32, data_type::u8>;
template struct simple_resampling_kernel_t<data_type::s8, data_type::f32>;
template struct simple_resampling_kernel_t<data_type::s8, data_type::bf16>;
template struct simple_resampling_kernel_t<data_type::s8, data_type::s32>;
template struct simple_resampling_kernel_t<data_type::s8, data_type::s8>;
template struct simple_resampling_kernel_t<data_type::s8, data_type::u8>;
template struct simple_resampling_kernel_t<data_type::u8, data_type::f32>;
template struct simple_resampling_kernel_t<data_type::u8, data_type::bf16>;
template struct simple_resampling_kernel_t<data_type::u8, data_type::s32>;
template struct simple_resampling_kernel_t<data_type::u8, data_type::s8>;
template struct simple_resampling_kernel_t<data_type::u8, data_type::u8>;

simple_resampling_base_t *create_simple_resampling(const resampling_pd_t *pd,
        const data_type_t src_dt, const data_type_t dst_dt) {
    using namespace data_type;

    switch (src_dt) {
        case f32:
            switch (dst_dt) {
                case f32: return new simple_resampling_kernel_t<f32, f32>(pd);
                case s32: return new simple_resampling_kernel_t<f32, s32>(pd);
                case bf16: return new simple_resampling_kernel_t<f32, bf16>(pd);
                case s8: return new simple_resampling_kernel_t<f32, s8>(pd);
                case u8: return new simple_resampling_kernel_t<f32, u8>(pd);
                default: break;
            }
        case s32:
            switch (dst_dt) {
                case f32: return new simple_resampling_kernel_t<s32, f32>(pd);
                case s32: return new simple_resampling_kernel_t<s32, s32>(pd);
                case bf16: return new simple_resampling_kernel_t<s32, bf16>(pd);
                case s8: return new simple_resampling_kernel_t<s32, s8>(pd);
                case u8: return new simple_resampling_kernel_t<s32, u8>(pd);
                default: break;
            }
        case bf16:
            switch (dst_dt) {
                case f32: return new simple_resampling_kernel_t<bf16, f32>(pd);
                case s32: return new simple_resampling_kernel_t<bf16, s32>(pd);
                case bf16:
                    return new simple_resampling_kernel_t<bf16, bf16>(pd);
                case s8: return new simple_resampling_kernel_t<bf16, s8>(pd);
                case u8: return new simple_resampling_kernel_t<bf16, u8>(pd);
                default: break;
            }
        case s8:
            switch (dst_dt) {
                case f32: return new simple_resampling_kernel_t<s8, f32>(pd);
                case s32: return new simple_resampling_kernel_t<s8, s32>(pd);
                case bf16: return new simple_resampling_kernel_t<s8, bf16>(pd);
                case s8: return new simple_resampling_kernel_t<s8, s8>(pd);
                case u8: return new simple_resampling_kernel_t<s8, u8>(pd);
                default: break;
            }
        case u8:
            switch (dst_dt) {
                case f32: return new simple_resampling_kernel_t<u8, f32>(pd);
                case s32: return new simple_resampling_kernel_t<u8, s32>(pd);
                case bf16: return new simple_resampling_kernel_t<u8, bf16>(pd);
                case s8: return new simple_resampling_kernel_t<u8, s8>(pd);
                case u8: return new simple_resampling_kernel_t<u8, u8>(pd);
                default: break;
            }
        default: break;
    }

    assert(!"Unsupported data type combination.");
    return nullptr;
}

} // namespace

simple_resampling_fwd_t::simple_resampling_fwd_t(const pd_t *apd)
    : primitive_t(apd), kernel_(nullptr) {}

status_t simple_resampling_fwd_t::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_,
            create_simple_resampling(pd(), pd()->src_md()->data_type,
                    pd()->dst_md()->data_type)));
    return kernel_->init();
}

status_t simple_resampling_fwd_t::execute(const exec_ctx_t &ctx) const {
    return kernel_->execute(ctx);
}

simple_resampling_bwd_t::simple_resampling_bwd_t(const pd_t *apd)
    : primitive_t(apd), kernel_(nullptr) {}

status_t simple_resampling_bwd_t::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_,
            create_simple_resampling(pd(), pd()->diff_dst_md()->data_type,
                    pd()->diff_src_md()->data_type)));
    return kernel_->init();
}

status_t simple_resampling_bwd_t::execute(const exec_ctx_t &ctx) const {
    return kernel_->execute(ctx);
}
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
