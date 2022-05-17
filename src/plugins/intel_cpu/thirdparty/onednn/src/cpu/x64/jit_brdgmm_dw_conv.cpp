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

#include "common/memory_tracking.hpp"
#include "common/utils.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_brdgmm_dw_conv.hpp"
#include <cpu/x64/cpu_isa_traits.hpp>

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

using namespace nstl;
using namespace data_type;

inline status_t init_tag(memory_desc_t &md, const memory_desc_wrapper &mdw,
        const format_tag_t tag_value, bool any_eligible) {

    format_tag_t tag;
    if (mdw.format_kind() == format_kind::any) {
        if (any_eligible) {
            CHECK(memory_desc_init_by_tag(md, tag_value));
            tag = tag_value;
        } else {
            tag = format_tag::undef;
        }
    } else {
        tag = mdw.matches_one_of_tag(tag_value);
    }

    if (tag != tag_value) return status::unimplemented;

    return status::success;
}

bool post_ops_ok(jit_brdgmm_conv_conf_t &jcp, const primitive_attr_t &attr,
        const memory_desc_wrapper &dst_d) {
    using namespace injector;

    const auto &post_ops = attr.post_ops_;

    return injector::post_ops_ok(post_ops_ok_args_t(get_max_cpu_isa(),
            {sum, eltwise, binary}, post_ops, &dst_d,
            false /*sum_at_pos_0_only*/, false /*sum_requires_scale_one*/,
            false /*sum_requires_zp_zero*/,
            {broadcasting_strategy_t::per_oc,
                    broadcasting_strategy_t::scalar}));
}

status_t brdgmm_dw_convolution_fwd_t::pd_t::init(engine_t *engine) {

    using skip_mask_t = primitive_attr_t::skip_mask_t;

    const auto &cd = *desc();
    const auto src_type = cd.src_desc.data_type;
    const auto wei_type = cd.weights_desc.data_type;
    const auto bia_type = cd.bias_desc.data_type;
    const auto dst_type = cd.dst_desc.data_type;

    // TODO: support s8s8 conv
    const bool is_f32 = everyone_is(f32, src_type, wei_type, dst_type);
    const bool is_int8 = one_of(src_type, u8) && wei_type == s8
            && one_of(dst_type, s32, f32, u8, s8);
    const bool is_bf16 = everyone_is(bf16, src_type, wei_type)
            && one_of(dst_type, bf16, f32);
    const auto isa = is_f32 ? avx512_core
                            : (is_int8 ? avx512_core_vnni : avx512_core_bf16);

    auto skip_mask = skip_mask_t::post_ops;
    if (is_int8) skip_mask |= skip_mask_t::oscale;

    bool ok = is_fwd() && set_default_alg_kind(alg_kind::convolution_direct)
            && one_of(true, is_f32, is_int8, is_bf16) && mayiuse(isa)
            && IMPLICATION(with_bias(),
                    ((one_of(src_type, u8, s8)
                             && one_of(bia_type, f32, s32, s8, u8))
                            || (one_of(src_type, bf16)
                                    && one_of(bia_type, f32, bf16))
                            || everyone_is(f32, src_type, bia_type)))
            && attr()->has_default_values(skip_mask) && !has_zero_dim_memory();
    if (!ok) return status::unimplemented;

    auto &jcp = jcp_;

    const memory_desc_wrapper src_d(&src_md_);
    const memory_desc_wrapper weights_d(&weights_md_);
    const memory_desc_wrapper dst_d(&dst_md_);
    const memory_desc_wrapper bias_d(&bias_md_);

    const int ndims = src_d.ndims();
    // Currently this kernel only supports 2D convolutions.
    if (ndims != 4) return status::unimplemented;
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    if (!with_groups) return status::unimplemented;
    // dilations are not supported
    if (cd.dilates[0] != 0 || cd.dilates[1] != 0) return status::unimplemented;

    jcp = zero<decltype(jcp)>();
    jcp.ngroups = weights_d.dims()[0];
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = dst_d.dims()[2];
    jcp.ow = dst_d.dims()[3];
    jcp.kh = weights_d.dims()[3];
    jcp.kw = weights_d.dims()[4];
    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, jcp.kh);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, jcp.kw);
    jcp.src_dt = cd.src_desc.data_type;
    jcp.dst_dt = cd.dst_desc.data_type;
    jcp.wei_dt = cd.weights_desc.data_type;
    jcp.with_bias = with_bias();
    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;

    if (!(everyone_is(1, jcp.ic, jcp.oc))) return status::unimplemented;

    const int simd_w = 16;
    const auto def_data_tag = format_tag::nhwc;
    const auto def_wei_tag = format_tag::hwioG16g;
    const bool any_eligible
            = (cd.prop_kind == prop_kind::forward_inference || is_int8);
    CHECK(init_tag(src_md_, src_d, def_data_tag, any_eligible));
    CHECK(init_tag(dst_md_, dst_d, def_data_tag, any_eligible));
    CHECK(init_tag(weights_md_, weights_d, def_wei_tag, true));

    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md_, format_tag::x));
    }

    CHECK(attr_.set_default_formats(dst_md()));
    if (!post_ops_ok(jcp, *attr(), dst_d)) return status::unimplemented;
    jcp.with_post_ops = attr()->post_ops_.len() > 0;

    jcp.isa = isa;
    jcp.nthr = dnnl_get_max_threads();
    jcp.src_dsz = types::data_type_size(jcp.src_dt);
    jcp.wei_dsz = types::data_type_size(jcp.wei_dt);
    jcp.bia_dsz
            = jcp.with_bias ? types::data_type_size(cd.bias_desc.data_type) : 0;
    jcp.dst_dsz = types::data_type_size(jcp.dst_dt);

    const auto &oscales = attr()->output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    jcp.ch_block = simd_w;
    jcp.nb_ch = div_up(jcp.ngroups, jcp.ch_block);

    // strd is only feasible for 1D (i.e., height dim is one)
    if (jcp.kh == 1) {
        jcp.batch_kind = brgemm_strd;
    } else if ((jcp.mb * jcp.oh) % jcp.nthr != 0) {
        jcp.batch_kind = brgemm_offs;
    } else {
        jcp.batch_kind = brgemm_addr;
    }

    // to avoid cache concurrent access from different threads
    size_t sc_size = sizeof(brgemm_batch_element_t);
    jcp.adjusted_batch_size
            = div_up(rnd_up(jcp.kh * jcp.kw * sc_size, 4096), sc_size);
    CHECK(init_brdgmm_conf());
    CHECK(init_scratchpad());
    return status::success;
}

status_t brdgmm_dw_convolution_fwd_t::pd_t::init_brdgmm_conf() {

    auto &jcp = jcp_;

    auto init_bcp = [&](int &idx, const int M, const int N) {
        const float alpha = 1.f;
        const float beta = 0.f;
        const int LDA = jcp.ngroups * jcp.stride_w;
        const int LDC = jcp.ngroups;
        const int LDD = jcp.ngroups;

        brgemm_attr_t brg_attr;
        brg_attr.max_bs = jcp.kw * jcp.kh;
        brg_attr.max_top_vpad = nstl::max(0, jcp.l_pad);
        brg_attr.max_bottom_vpad = nstl::max(0, jcp.r_pad);

        // only needed for strd batch_kind
        const brgemm_strides_t strides
                = {static_cast<dim_t>(jcp.src_dsz) * jcp.ngroups,
                        static_cast<dim_t>(jcp.wei_dsz)
                                * rnd_up(jcp.ngroups, jcp.ch_block)};

        auto &bcp = bcps_[idx];
        CHECK(brdgmm_desc_init(&bcp, jcp.isa, jcp.batch_kind, jcp.src_dt,
                jcp.wei_dt, false /*transA*/, brgemm_row_major, alpha, beta,
                LDA, LDC, M, N, &strides));
        CHECK(brgemm_desc_set_attr(&bcp, brg_attr));
        CHECK(brgemm_desc_set_postops(&bcp, attr(), dst_md(), LDD, jcp.bia_dt));
        ++idx;
        return status::success;
    };

    bcps_.resize(1);
    jcp.ow_block = jcp.ow;
    jcp.nb_ow = 1;
    jcp.nb_ch_blocking = jcp.ngroups;
    jcp.chb_tail = 0;
    int ker_idx = 0;
    CHECK(init_bcp(ker_idx, jcp.ow, jcp.ngroups)); // default full row kernel.

    if ((jcp.mb * jcp.oh) % jcp.nthr != 0) {
        // determine ow_block
        {
            const size_t work_amount = jcp.mb * jcp.oh * jcp.ow;
            if (work_amount % jcp.nthr == 0) {
                const size_t work_per_thr = div_up(work_amount, jcp.nthr);
                const size_t ow_tail_block
                        = (work_per_thr / jcp.nb_ch) % jcp.ow;
                if (ow_tail_block && (jcp.ow % ow_tail_block == 0))
                    jcp.ow_block = ow_tail_block;
                else {
                    jcp.ow_block = jcp.ow;
                }
            } else {
                jcp.ow_block = nstl::min(6, jcp.ow);
            }
            jcp.ow_tail = jcp.ow % jcp.ow_block;
        }
        jcp.nb_ow = div_up(jcp.ow, jcp.ow_block);

        // determine nb_ch_block
        {
            const size_t work_amount = jcp.mb * jcp.nb_ch * jcp.oh * jcp.nb_ow;
            if (work_amount % jcp.nthr == 0) {
                const size_t work_per_thr = div_up(work_amount, jcp.nthr);
                const size_t ch_tail_block = work_per_thr % jcp.nb_ch;
                if (ch_tail_block && (jcp.nb_ch % ch_tail_block == 0))
                    jcp.nb_ch_blocking = ch_tail_block * jcp.ch_block;
                else
                    jcp.nb_ch_blocking = jcp.ngroups;
            } else {
                jcp.nb_ch_blocking = nstl::min(4 * jcp.ch_block, jcp.ngroups);
            }
            jcp.chb_tail = jcp.ngroups % jcp.nb_ch_blocking;
        }

        const int n_owb_kernels = std::ceil(log2(jcp.nb_ow));
        const int num_kernels = 1 /*full ow*/ + n_owb_kernels
                + (jcp.chb_tail != 0) + (jcp.nb_ch_blocking != jcp.ngroups)
                + (jcp.ow_tail != 0);
        bcps_.resize(num_kernels);

        for (int i = 0; i < n_owb_kernels; ++i) {
            CHECK(init_bcp(ker_idx, jcp.ow_block * (1 << i), jcp.ngroups));
        }

        if (jcp.chb_tail) {
            jcp.chb_tail_idx = ker_idx;
            CHECK(init_bcp(ker_idx, jcp.ow_block, jcp.chb_tail));
        }

        if (jcp.ow_tail) {
            jcp.ow_tail_idx = ker_idx;
            CHECK(init_bcp(ker_idx, jcp.ow_tail, jcp.ngroups));
        }

        if (jcp.nb_ch_blocking != jcp.ngroups) {
            jcp.nb_ch_blocking_idx = ker_idx;
            CHECK(init_bcp(ker_idx, jcp.ow_block, jcp.nb_ch_blocking));
        }
        assert(num_kernels == ker_idx);
    }

    return status::success;
}

status_t brdgmm_dw_convolution_fwd_t::pd_t::init_scratchpad() {
    const auto &jcp = jcp_;
    auto scratchpad = scratchpad_registry().registrar();

    scratchpad.book(key_brgemm_primitive_batch,
            static_cast<size_t>(jcp.nthr) * jcp.adjusted_batch_size,
            sizeof(brgemm_batch_element_t), 64);
    return status::success;
}

status_t brdgmm_dw_convolution_fwd_t::init(engine_t *engine) {
    const auto &bcps = pd()->bcps_;
    brdgmm_kernels_.resize(bcps.size());

    for (size_t idx = 0; idx < bcps.size(); ++idx) {
        const auto &bcp = bcps[idx];
        if (bcp.bcast_dim * bcp.load_dim /* M*N */ == 0) continue;
        brgemm_kernel_t *brg_kernel = nullptr;
        CHECK(brgemm_kernel_create(&brg_kernel, pd()->bcps_[idx]));
        CHECK(safe_ptr_assign(brdgmm_kernels_[idx], brg_kernel));
    }

    return status::success;
}

status_t brdgmm_dw_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {

    const char *const __restrict src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    const char *const __restrict weights
            = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
    const char *const __restrict bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    char *const __restrict dst = CTX_OUT_MEM(const char *, DNNL_ARG_DST);
    const float *oscales = pd()->attr()->output_scales_.scales_;
    const memory_tracking::grantor_t scratchpad = ctx.get_scratchpad_grantor();
    brgemm_batch_element_t *const __restrict brg_batch_global
            = scratchpad.template get<brgemm_batch_element_t>(
                    key_brgemm_primitive_batch);
    const std::vector<const void *> post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(
                    pd()->attr()->post_ops_, ctx);

    const auto &jcp = pd()->jcp_;
    const int chb_step = jcp.nb_ch_blocking;
    const int chb_work = div_up(jcp.ngroups, chb_step);
    const int ow_step = jcp.ow_block;
    const int work_amount = jcp.mb * jcp.oh * jcp.nb_ow * chb_work;

    const size_t src_ch_stride = jcp.src_dsz;
    const size_t src_w_stride = jcp.ngroups * jcp.src_dsz;
    const size_t src_h_stride = jcp.ngroups * jcp.iw * jcp.src_dsz;
    const size_t src_mb_stride = jcp.ngroups * jcp.iw * jcp.ih * jcp.src_dsz;
    const size_t wei_ch_stride = jcp.wei_dsz;
    const size_t wei_w_stride = rnd_up(jcp.ngroups, jcp.ch_block) * jcp.wei_dsz;
    const size_t wei_h_stride = wei_w_stride * jcp.kw;
    const size_t dst_ch_stride = jcp.dst_dsz;
    const size_t dst_w_stride = jcp.ngroups * jcp.dst_dsz;
    const size_t dst_h_stride = jcp.ngroups * jcp.ow * jcp.dst_dsz;
    const size_t dst_mb_stride = jcp.ngroups * jcp.ow * jcp.oh * jcp.dst_dsz;

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        int n {0}, chb {0}, oh {0}, owb {0};

        auto iwork = start;
        brgemm_batch_element_t *const __restrict brg_batch = brg_batch_global
                + static_cast<size_t>(ithr) * jcp.adjusted_batch_size;
        const brgemm_kernel_t *kernel = nullptr;
        const brgemm_kernel_t *kernel_chb_tail
                = brdgmm_kernels_[jcp.chb_tail_idx].get();
        brgemm_post_ops_data_t post_ops_data;
        post_ops_data.binary_post_ops_rhs = post_ops_binary_rhs_arg_vec.data();

        while (iwork < end) {
            nd_iterator_init(iwork, n, jcp.mb, oh, jcp.oh, owb, jcp.nb_ow, chb,
                    chb_work);
            const bool is_m_tail = jcp.ow_tail != 0 && (owb + 1 == jcp.nb_ow);
            const bool is_n_tail = jcp.chb_tail != 0 && (chb + 1 == chb_work);
            if (is_m_tail && chb != 0) {
                // the tail ow_block is not split btw threads to reduce the
                // number of kernels.
                utils::nd_iterator_jump(iwork, end, n, jcp.mb, oh, jcp.oh, owb,
                        jcp.nb_ow, chb, chb_work);
                continue;
            }

            const auto rem_work = end - iwork;
            const int rem_row_owb
                    = saturate(1, jcp.nb_ow - owb, rem_work / chb_work);
            int cur_n_owb = 1;
            int ker_idx = 0;
            if (is_n_tail) {
                ker_idx = jcp.chb_tail_idx;
            } else if (is_m_tail) {
                ker_idx = jcp.ow_tail_idx;
            } else if (chb != 0 || rem_work < chb_work) {
                ker_idx = jcp.nb_ch_blocking_idx;
            } else if (rem_row_owb == jcp.nb_ow) {
                ker_idx = 0;
                cur_n_owb = jcp.nb_ow;
            } else {
                // The ow_tail kernel is processed alone, subtract if it exists.
                const int log_rem_owb = log2(rem_row_owb
                        - (owb + rem_row_owb >= jcp.nb_ow)
                                * (jcp.ow_tail != 0));
                cur_n_owb = (1 << log_rem_owb);
                ker_idx = log_rem_owb + 1; // add 1 as 0th is full row.
            }

            kernel = brdgmm_kernels_[ker_idx].get();

            int ch = chb * chb_step;
            const int ow = owb * ow_step;
            auto *ptr_A = src;
            auto *ptr_B = weights;
            int bs = 0;
            for (int kh = 0; kh < jcp.kh; ++kh) {
                for (int kw = 0; kw < jcp.kw; ++kw) {
                    const int ih = (oh * jcp.stride_h - jcp.t_pad) + kh;
                    if (ih < 0 || ih >= jcp.ih) continue;
                    const int iw_s = ow * jcp.stride_w - jcp.l_pad + kw;
                    const int ow_e
                            = nstl::min(jcp.ow, ow + cur_n_owb * jcp.ow_block)
                            - 1;
                    const int iw_e = ow_e * jcp.stride_w - jcp.l_pad + kw;
                    auto &batch = brg_batch[bs];
                    batch.vvpad.top = nstl::max(0, div_up(-iw_s, jcp.stride_w));
                    batch.vvpad.bottom = nstl::max<dim_t>(
                            0, div_up(iw_e - (jcp.iw - 1), jcp.stride_w));
                    const dim_t offs_A = n * src_mb_stride + ih * src_h_stride
                            + iw_s * src_w_stride + ch * src_ch_stride;
                    const dim_t offs_B = kh * wei_h_stride + kw * wei_w_stride
                            + ch * wei_ch_stride;
                    if (jcp.batch_kind == brgemm_offs) {
                        batch.offset.A = offs_A;
                        batch.offset.B = offs_B;
                    } else if (jcp.batch_kind == brgemm_addr) {
                        batch.ptr.A = src + offs_A;
                        batch.ptr.B = weights + offs_B;
                    } else {
                        assert(jcp.batch_kind == brgemm_strd);
                        if (bs == 0) {
                            ptr_A = src + offs_A;
                            ptr_B = weights + offs_B;
                        }
                    }
                    ++bs;
                }
            }
            auto ptr_C = dst + n * dst_mb_stride + oh * dst_h_stride
                    + ow * dst_w_stride + ch * dst_ch_stride;
            const int rem_chb_work = chb_work - chb;
            int chb_loop_work = is_m_tail || (chb == 0 && rem_work >= chb_work)
                    ? 1 // Compute entire chb_work in single jit call
                    : nstl::min(rem_work, rem_chb_work);
            iwork += cur_n_owb * nstl::min(rem_work, rem_chb_work);

            while (chb_loop_work) {
                // brgemm_offs and brgemm_strd mode enables us to run this loop,
                // without changing brg_batch elements.
                assert(IMPLICATION(chb != 0,
                        one_of(jcp.batch_kind, brgemm_offs, brgemm_strd)));
                post_ops_data.bias = bias + ch * jcp.bia_dsz;
                post_ops_data.scales = &oscales[jcp.is_oc_scale * ch];
                post_ops_data.oc_logical_off = ch;
                brgemm_kernel_execute_postops(kernel, bs, ptr_A, ptr_B,
                        brg_batch, ptr_C, ptr_C, post_ops_data,
                        nullptr /*scratch*/);
                ++chb;
                if (jcp.chb_tail != 0 && chb + 1 == chb_work)
                    kernel = kernel_chb_tail;
                ch += chb_step;
                ptr_A += chb_step * src_ch_stride;
                ptr_B += chb_step * wei_ch_stride;
                ptr_C += chb_step * dst_ch_stride;
                --chb_loop_work;
            }
        }
    });
    return status::success;
}
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
