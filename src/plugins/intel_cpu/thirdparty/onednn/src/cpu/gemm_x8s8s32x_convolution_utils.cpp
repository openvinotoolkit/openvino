/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <algorithm>
#include <cstdlib>
#include <memory>

#include "common/math_utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"
#include "cpu/ref_io_helper.hpp"
#include "cpu/simple_q10n.hpp"

#if DNNL_X64
#include "cpu/x64/jit_gemm_x8s8s32x_convolution_utils.hpp"
#endif

#include "cpu/gemm_x8s8s32x_convolution_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace gemm_x8s8s32x_convolution_utils {

template <typename dst_data_t>
struct ref_pp_ker_t : pp_ker_t {
    ref_pp_ker_t(const convolution_pd_t *pd, const conv_gemm_conf_t &jcp)
        : pp_ker_t(pd, jcp) {
        for (int i = 0; i < post_ops_.len(); i++) {
            auto &post_op = post_ops_.entry_[i];
            if (post_op.is_eltwise()) {
                ref_eltwise_injectors_.push_back(new ref_eltwise_scalar_fwd_t(post_op.eltwise));
            } else if (post_op.is_depthwise()) {
                ref_depthwise_injectors_.push_back(new ref_depthwise_scalar_fwd_t(
                        post_op.depthwise.alg));
            }
        }
    }
    ~ref_pp_ker_t() {
        for (auto impl : ref_eltwise_injectors_)
            delete impl;
        ref_eltwise_injectors_.clear();
        for (auto impl : ref_depthwise_injectors_)
            delete impl;
        ref_depthwise_injectors_.clear();
    }

    using acc_data_t = pp_ker_t::acc_data_t;

    void operator()(void *dst, acc_data_t *acc, const char *bias,
            const float *scales, float sum_scale, float signed_scale, int g,
            size_t start, size_t end, const zero_point_call_params_t &zp,
            const void *post_ops_binary_rhs_arg_vec, const void *dst_orig,
            const exec_ctx_t &ctx, const memory_desc_t &dst_md,
            const single_gemm_conv_chunk_desc_t &chunk_desc) const override;

private:
    nstl::vector<ref_eltwise_scalar_fwd_t*> ref_eltwise_injectors_;
    nstl::vector<ref_depthwise_scalar_fwd_t*> ref_depthwise_injectors_;
};

template <typename dst_data_t>
void ref_pp_ker_t<dst_data_t>::operator()(void *void_dst, acc_data_t *acc, const char *bias, const float *scales, float sum_scale,
        float signed_scale, int g, size_t start, size_t end,
        const zero_point_call_params_t &zp,
        const void * post_ops_binary_rhs_arg_vec,
        const void * /* dst_orig */, const exec_ctx_t &ctx,
        const memory_desc_t &dst_md,
        const single_gemm_conv_chunk_desc_t &chunk_desc) const {

    if (end <= start) return;

    assert(data_traits<dst_data_t>::data_type == dst_data_type_);
    dst_data_t *dst = (dst_data_t *)void_dst;

    const size_t first_oc = start % OC_;
    const size_t last_oc = (end - 1) % OC_;
    const size_t first_os = start / OC_;
    const size_t last_os = (end - 1) / OC_;
    if (post_ops_.len() == 0) {
        for (size_t os = first_os; os <= last_os; os++) {
            const size_t start_oc = (os == first_os) ? first_oc : 0;
            const size_t end_oc = (os == last_os) ? last_oc : OC_ - 1;
            for (size_t oc = start_oc; oc <= end_oc; oc++) {
                const size_t acc_off = os * jcp_.oc + oc;
                const size_t dst_off = os * dst_os_stride_ + oc;

                float d = (float) (acc[acc_off]);
                if (jcp_.signed_input) d *= signed_scale;

                if (do_bias_)
                    d += math::get_bias(bias, g * jcp_.oc + oc, bias_data_type_);

                d *= scales[(g * jcp_.oc + oc) * scale_idx_mult_];
                dst[dst_off] = qz_a1b0<float, dst_data_t>()(d);
            }
        }
    } else {
        float* acc_fp = reinterpret_cast<float*>(acc);

        auto load = [&](int idx, size_t oc, size_t os, size_t acc_off, size_t dst_off) {
            float d;
            if (idx == 0) {
                d = (float) (acc[acc_off]);

                if (jcp_.signed_input)
                    d *= signed_scale;

                if (do_bias_)
                    d += math::get_bias(bias, g * jcp_.oc + oc,
                                        bias_data_type_);

                d *= scales[(g * jcp_.oc + oc) * scale_idx_mult_];
            } else {
                d = acc_fp[acc_off];
            }

            return d;
        };

        auto store = [&](int idx, float d, size_t acc_off, size_t dst_off) {
            if (idx == post_ops_.len() - 1)
                dst[dst_off] = qz_a1b0<float, dst_data_t>()(d);
            else
                acc_fp[acc_off] = d;
        };

        auto post_ops_data_ptrs = reinterpret_cast<const float* const*>(post_ops_binary_rhs_arg_vec);
        std::size_t post_ops_data_idx = 0;
        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        for (int i = 0; i < post_ops_.len(); i++) {
            auto &post_op = post_ops_.entry_[i];
            if (post_op.is_eltwise()) {
                for (size_t os = first_os; os <= last_os; os++) {
                    const size_t start_oc = (os == first_os) ? first_oc : 0;
                    const size_t end_oc = (os == last_os) ? last_oc : OC_ - 1;
                    for (size_t oc = start_oc; oc <= end_oc; oc++) {
                        const size_t acc_off = os * jcp_.oc + oc;
                        const size_t dst_off = os * this->dst_os_stride_ + oc;

                        float d = load(i, oc, os, acc_off, dst_off);

                        d = ref_eltwise_injectors_[eltwise_inj_idx]->compute_scalar(d);

                        store(i, d, acc_off, dst_off);
                    }
                }
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                for (size_t os = first_os; os <= last_os; os++) {
                    const size_t start_oc = (os == first_os) ? first_oc : 0;
                    const size_t end_oc = (os == last_os) ? last_oc : OC_ - 1;
                    for (size_t oc = start_oc; oc <= end_oc; oc++) {
                        const size_t acc_off = os * jcp_.oc + oc;
                        const size_t dst_off = os * this->dst_os_stride_ + oc;

                        auto depthwise_base = post_ops_data_ptrs[post_ops_data_idx];
                        auto depthwise_weights = depthwise_base + post_op.depthwise.offset[post_op.depthwise.scales];
                        auto depthwise_bias = depthwise_base + post_op.depthwise.offset[post_op.depthwise.shifts];

                        float d = load(i, oc, os, acc_off, dst_off);

                        d = ref_depthwise_injectors_[depthwise_inj_idx]->compute_scalar(d, depthwise_weights + g * jcp_.oc + oc,
                                                                                        depthwise_bias + g * jcp_.oc + oc);

                        store(i, d, acc_off, dst_off);

                    }
                }
                post_ops_data_idx++;
                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                for (size_t os = first_os; os <= last_os; os++) {
                    const size_t start_oc = (os == first_os) ? first_oc : 0;
                    const size_t end_oc = (os == last_os) ? last_oc : OC_ - 1;
                    for (size_t oc = start_oc; oc <= end_oc; oc++) {
                        const size_t acc_off = os * jcp_.oc + oc;
                        const size_t dst_off = os * this->dst_os_stride_ + oc;

                        auto quant = post_op.quantization;
                        auto quantization_base = post_ops_data_ptrs[post_ops_data_idx];
                        auto pcl = quantization_base + post_op.quantization.offset[quant.crop_low];
                        auto pch = quantization_base + post_op.quantization.offset[quant.crop_high];
                        auto pisc = quantization_base + post_op.quantization.offset[quant.inp_scale];
                        auto pish = quantization_base + post_op.quantization.offset[quant.inp_shift];
                        auto posc = quantization_base + post_op.quantization.offset[quant.output_scale];
                        auto posh = quantization_base + post_op.quantization.offset[quant.output_shift];

                        float d = load(i, oc, os, acc_off, dst_off);

                        int cl_idx = !quant.per_channel[quant.crop_low] ? 0 : g * jcp_.oc + oc;
                        int ch_idx = !quant.per_channel[quant.crop_high] ? 0 : g * jcp_.oc + oc;
                        int isc_idx = !quant.per_channel[quant.inp_scale] ? 0 : g * jcp_.oc + oc;
                        int ish_idx = !quant.per_channel[quant.inp_shift] ? 0 : g * jcp_.oc + oc;
                        int osc_idx = !quant.per_channel[quant.output_scale] ? 0 : g * jcp_.oc + oc;
                        int osh_idx = !quant.per_channel[quant.output_shift] ? 0 : g * jcp_.oc + oc;

                        d = nstl::min(pch[ch_idx], nstl::max(pcl[cl_idx], d));
                        d = d * pisc[isc_idx] + pish[ish_idx];
                        d = roundf(d);
                        d = d * posc[osc_idx] + posh[osh_idx];

                        store(i, d, acc_off, dst_off);

                    }
                }
                post_ops_data_idx++;
            } else if (post_op.is_sum()) {
                for (size_t os = first_os; os <= last_os; os++) {
                    const size_t start_oc = (os == first_os) ? first_oc : 0;
                    const size_t end_oc = (os == last_os) ? last_oc : OC_ - 1;
                    for (size_t oc = start_oc; oc <= end_oc; oc++) {
                        const size_t acc_off = os * jcp_.oc + oc;
                        const size_t dst_off = os * this->dst_os_stride_ + oc;

                        float d = load(i, oc, os, acc_off, dst_off);

                        d += post_op.sum.scale * math::get_sum((char *) dst, dst_off, post_op.sum.dt);

                        store(i, d, acc_off, dst_off);
                    }
                }
            }
        }
    }
}

// Interface section

pp_ker_t::pp_ker_t(const convolution_pd_t *pd, const conv_gemm_conf_t &jcp)
    : jcp_(jcp)
    , post_ops_(pd->attr()->post_ops_)
    , OC_(jcp_.oc)
{
    const auto dst_md = memory_desc_wrapper(pd->dst_md());

    dst_os_stride_ = dst_md.blocking_desc().strides[pd->ndims() - 1];
    dst_data_type_ = dst_md.data_type();

    do_scale_ = !pd->attr()->output_scales_.has_default_values();
    if (do_scale_) {
        scale_idx_mult_ = (pd->attr()->output_scales_.mask_ == (1 << 1));
    }

    do_bias_ = pd->with_bias();
    if (do_bias_) {
        bias_data_type_ = pd->desc()->bias_desc.data_type;
        assert(bias_data_type_ != data_type::undef);
    }
}

pp_ker_t *pp_ker_t::create(
        const convolution_pd_t *pd, const conv_gemm_conf_t &jcp) {
#if DNNL_X64
    auto *res
            = x64::gemm_x8s8s32x_convolution_utils::jit_pp_ker_create(pd, jcp);
    if (res) return res;
#endif
    switch (pd->dst_md()->data_type) {
        case data_type::f32: return new ref_pp_ker_t<float>(pd, jcp);
        case data_type::s32: return new ref_pp_ker_t<int32_t>(pd, jcp);
        case data_type::s8: return new ref_pp_ker_t<int8_t>(pd, jcp);
        case data_type::u8: return new ref_pp_ker_t<uint8_t>(pd, jcp);
        default: assert(!"unexpected data type");
    }
    return nullptr;
}

} // namespace gemm_x8s8s32x_convolution_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl
