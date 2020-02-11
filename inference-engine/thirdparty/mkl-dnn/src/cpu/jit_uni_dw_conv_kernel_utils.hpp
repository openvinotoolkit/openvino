
/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef JIT_UNI_DW_CONVOLUTION_UTILS_HPP
#define JIT_UNI_DW_CONVOLUTION_UTILS_HPP

#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_eltwise.hpp"

#include "jit_avx512_core_bf16_dw_conv_kernel.hpp"
#include "jit_uni_dw_conv_kernel_f32.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;

template <cpu_isa_t isa, data_type_t kernel_dt>
struct jit_uni_dw_conv_fwd_kernel {

    jit_uni_dw_conv_fwd_kernel(jit_conv_conf_t ajcp, const primitive_attr_t &attr)
        : jit_ker(nullptr), ker_(nullptr) {
        ker_ = new jit_kernel_t(ajcp, attr);
        jit_ker = ker_->jit_ker;
    }
    ~jit_uni_dw_conv_fwd_kernel() { delete ker_; }

    static bool post_ops_ok(jit_conv_conf_t &jcp, const primitive_attr_t &attr, bool is_bf16);

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, const primitive_attr_t &attr);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    void (*jit_ker)(jit_conv_call_s *);

private:
    using jit_kernel_t = typename utils::conditional<isa == avx512_core
                    && kernel_dt == data_type::bf16,
            jit_avx512_dw_conv_fwd_kernel_bf16,
            jit_uni_dw_conv_fwd_kernel_f32<isa>>::type;
    jit_kernel_t *ker_;
};

template <cpu_isa_t isa, data_type_t kernel_dt>
bool jit_uni_dw_conv_fwd_kernel<isa, kernel_dt>::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr, bool is_bf16) {
    const auto &p = attr.post_ops_;

    if (is_bf16) {
        auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };
        auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(); };

        switch (p.len_) {
            case 0: return true; // no post_ops
            case 1: return is_eltwise(0) || is_sum(0); // sum OR eltwise
            case 2: return is_sum(0) && is_eltwise(1); // sum -> eltwise
            default: return false;
        }
    } else {
        auto all_post_ops_supported = [&]() {
            bool ok = true;

            for (int i = 0; i < p.len_; i++) {
                ok = ok && utils::one_of(p.entry_[i].kind, primitive_kind::sum, primitive_kind::eltwise, primitive_kind::depthwise, primitive_kind::quantization);
            }
            return ok;
        };
        auto contain = [&](mkldnn::impl::primitive_kind_t kind) { return p.find(kind) != -1; };
        auto position = [&](mkldnn::impl::primitive_kind_t kind) { return p.find(kind); };
        auto count = [&](mkldnn::impl::primitive_kind_t kind) { return p.count(kind); };

        return all_post_ops_supported() &&
               count(primitive_kind::sum) <= 1 &&
               IMPLICATION(contain(primitive_kind::sum), position(primitive_kind::sum) == 0);
    }

    return false;
}

template <cpu_isa_t isa, data_type_t kernel_dt>
status_t jit_uni_dw_conv_fwd_kernel<isa, kernel_dt>::init_conf(
        jit_conv_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const primitive_attr_t &attr) {

    jcp.dst_dt = cd.dst_desc.data_type;
    const bool is_bf16 = src_d.data_type() == data_type::bf16;

    if (!mayiuse(isa) || (is_bf16 && !mayiuse(avx512_core)))
        return status::unimplemented;

    const int simd_w = one_of(isa, avx512_common, avx512_core) ? 16 : 8;

    jcp.prop_kind = cd.prop_kind;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    if (!with_groups) return status::unimplemented;

    jcp.ngroups = weights_d.dims()[0];
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1];
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1];

    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = dst_d.dims()[2];
    jcp.ow = dst_d.dims()[3];

    jcp.kh = weights_d.dims()[3];
    jcp.kw = weights_d.dims()[4];

    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.b_pad = cd.padding[1][0];
    jcp.r_pad = cd.padding[1][1];

    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];

    jcp.src_fmt = src_d.format();
    jcp.with_bias = cd.bias_desc.format != memory_format::undef;

    if (!post_ops_ok(jcp, attr, is_bf16))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise)
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;

    bool ok_to_pad_channels = true
        && jcp.oc == jcp.ngroups
        && jcp.ic == jcp.ngroups
        && one_of(isa, avx512_common, avx512_core, avx2);
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.oc, simd_w);
        jcp.ngroups = rnd_up(jcp.ngroups, simd_w);
    }

    auto desired_act_fmt
            = one_of(isa, avx512_common, avx512_core) ? nChw16c : nChw8c;
    auto desired_wei_fmt
            = one_of(isa, avx512_common, avx512_core) ? Goihw16g : Goihw8g;

    bool args_ok = true
        && jcp.oc == jcp.ngroups
        && jcp.ic == jcp.ngroups
        && jcp.ngroups % simd_w == 0
        && src_d.format() == desired_act_fmt
        && weights_d.format() == desired_wei_fmt
        && one_of(cd.bias_desc.format, memory_format::undef, any, x)
        && dst_d.format() == desired_act_fmt
        && jcp.ic <= src_d.blocking_desc().padding_dims[1]
        && jcp.oc <= dst_d.blocking_desc().padding_dims[1]
        && jcp.ngroups <= weights_d.blocking_desc().padding_dims[0];
    if (!args_ok) return status::unimplemented;

    jcp.is_cpx = (mayiuse(avx512_core_bf16)) ? true : false;

    jcp.typesize_out = jcp.dst_dt == data_type::bf16 ? sizeof(mkldnn_bfloat16_t)
                                                     : sizeof(float);
    jcp.typesize_in = src_d.data_type() == data_type::bf16
            ? sizeof(mkldnn_bfloat16_t)
            : sizeof(float);

    jcp.ur_w = is_bf16 ? (jcp.is_cpx ? 6 : 4)
                       : isa == avx512_common ? 6 : isa == avx2 ? 4 : 3;

    jcp.ch_block = simd_w;
    jcp.nb_ch = jcp.oc / jcp.ch_block;
    jcp.nb_ch_blocking
            = one_of(isa, avx512_common, avx512_core) ? 4 : isa == avx2 ? 3 : 2;
    if (jcp.nb_ch < jcp.nb_ch_blocking)
        jcp.nb_ch_blocking = jcp.nb_ch;

    return status::success;
}

template <cpu_isa_t isa, data_type_t kernel_dt>
void jit_uni_dw_conv_fwd_kernel<isa, kernel_dt>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    if (jcp.with_bias && jcp.oc_without_padding != jcp.oc)
        scratchpad.book(key_conv_padded_bias, sizeof(float) * jcp.oc);
}

template struct jit_uni_dw_conv_fwd_kernel<avx512_core, data_type::bf16>;
template struct jit_uni_dw_conv_fwd_kernel<avx512_common, data_type::f32>;
template struct jit_uni_dw_conv_fwd_kernel<avx2, data_type::f32>;
template struct jit_uni_dw_conv_fwd_kernel<sse42, data_type::f32>;

template <cpu_isa_t isa, data_type_t kernel_dt>
struct jit_uni_dw_conv_bwd_data_kernel {

    jit_uni_dw_conv_bwd_data_kernel(jit_conv_conf_t ajcp)
        : jit_ker(nullptr), ker_(nullptr) {
        ker_ = new jit_kernel_t(ajcp);
        jit_ker = ker_->jit_ker;
    }
    ~jit_uni_dw_conv_bwd_data_kernel(){
        delete ker_;
    }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &diff_src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &diff_dst_d);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    void (*jit_ker)(jit_conv_call_s *);

private:
    using jit_kernel_t = typename utils::conditional<isa == avx512_core
                    && kernel_dt == data_type::bf16,
            jit_avx512_dw_conv_bwd_data_kernel_bf16,
            jit_uni_dw_conv_bwd_data_kernel_f32<isa>>::type;
    jit_kernel_t *ker_;
};

template <cpu_isa_t isa, data_type_t kernel_dt>
status_t jit_uni_dw_conv_bwd_data_kernel<isa, kernel_dt>::init_conf(
        jit_conv_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d) {

    jcp.dsrc_dt = cd.diff_src_desc.data_type;
    const bool is_bf16 = diff_dst_d.data_type() == data_type::bf16;

    if (!mayiuse(isa) || (is_bf16 && !mayiuse(avx512_core)))
        return status::unimplemented;

    const int simd_w = one_of(isa, avx512_common, avx512_core) ? 16 : 8;

    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;
    if (!with_groups) return status::unimplemented;

    jcp.ngroups = weights_d.dims()[0];
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1];
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = diff_src_d.dims()[1];

    jcp.ih = diff_src_d.dims()[2];
    jcp.iw = diff_src_d.dims()[3];
    jcp.oh = diff_dst_d.dims()[2];
    jcp.ow = diff_dst_d.dims()[3];

    jcp.kh = weights_d.dims()[3];
    jcp.kw = weights_d.dims()[4];

    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.b_pad = cd.padding[1][0];
    jcp.r_pad = cd.padding[1][1];

    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;

    jcp.src_fmt = diff_src_d.format();

    bool ok_to_pad_channels = true
        && jcp.oc == jcp.ngroups
        && jcp.ic == jcp.ngroups
        && one_of(isa, avx512_common, avx512_core, avx2, sse42);
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.oc, simd_w);
        jcp.ngroups = rnd_up(jcp.ngroups, simd_w);
    }

    auto desired_act_fmt
            = one_of(isa, avx512_common, avx512_core) ? nChw16c : nChw8c;
    auto desired_wei_fmt
            = one_of(isa, avx512_common, avx512_core) ? Goihw16g : Goihw8g;

    bool args_ok = true
        && jcp.oc == jcp.ngroups
        && jcp.ic == jcp.ngroups
        && jcp.ngroups % simd_w == 0
        && jcp.dilate_h == 0
        && jcp.dilate_w == 0
        && diff_src_d.format() == desired_act_fmt
        && weights_d.format() == desired_wei_fmt
        && diff_dst_d.format() == desired_act_fmt
        && jcp.oh == (jcp.ihp - jcp.kh) / jcp.stride_h + 1
        && jcp.ow == (jcp.iwp - jcp.kw) / jcp.stride_w + 1
        && jcp.ic <= diff_src_d.blocking_desc().padding_dims[1]
        && jcp.oc <= diff_dst_d.blocking_desc().padding_dims[1]
        && jcp.ngroups <= weights_d.blocking_desc().padding_dims[0];
    if (!args_ok) return status::unimplemented;

    jcp.is_cpx = (mayiuse(avx512_core_bf16)) ? true : false;

    jcp.typesize_out = diff_src_d.data_type() == data_type::bf16
            ? sizeof(mkldnn_bfloat16_t)
            : sizeof(float);
    jcp.typesize_in = diff_dst_d.data_type() == data_type::bf16
            ? sizeof(mkldnn_bfloat16_t)
            : sizeof(float);

    jcp.ur_w = is_bf16 ? (jcp.is_cpx ? 6 : 4)
                       : isa == avx512_common ? 6 : isa == avx2 ? 4 : 3;

    jcp.ch_block = simd_w;
    jcp.nb_ch = jcp.ic / jcp.ch_block;
    jcp.nb_ch_blocking
            = one_of(isa, avx512_common, avx512_core) ? 4 : isa == avx2 ? 3 : 2;
    if (jcp.nb_ch < jcp.nb_ch_blocking)
        jcp.nb_ch_blocking = jcp.nb_ch;

    return status::success;
}

template <cpu_isa_t isa, data_type_t kernel_dt>
void jit_uni_dw_conv_bwd_data_kernel<isa, kernel_dt>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    UNUSED(scratchpad);
    UNUSED(jcp);
}

template struct jit_uni_dw_conv_bwd_data_kernel<avx512_core, data_type::bf16>;
template struct jit_uni_dw_conv_bwd_data_kernel<avx512_common, data_type::f32>;
template struct jit_uni_dw_conv_bwd_data_kernel<avx2, data_type::f32>;
template struct jit_uni_dw_conv_bwd_data_kernel<sse42, data_type::f32>;

template <cpu_isa_t isa, data_type_t kernel_dt>
struct jit_uni_dw_conv_bwd_weights_kernel {

    jit_uni_dw_conv_bwd_weights_kernel(jit_conv_conf_t ajcp)
        : jit_ker(nullptr), ker_(nullptr) {
        ker_ = new jit_kernel_t(ajcp);
        jit_ker = ker_->jit_ker;
    }

    ~jit_uni_dw_conv_bwd_weights_kernel() { delete ker_; }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &diff_weights_d,
            const memory_desc_wrapper &diff_dst_d, int nthreads);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    static void balance(jit_conv_conf_t &jcp, int nthreads);

    void (*jit_ker)(jit_dw_conv_call_s *);

private:
    using jit_kernel_t = typename utils::conditional<isa == avx512_core
                    && kernel_dt == data_type::bf16,
            jit_avx512_dw_conv_bwd_weights_kernel_bf16,
            jit_uni_dw_conv_bwd_weights_kernel_f32<isa>>::type;
    jit_kernel_t *ker_;
};

template <cpu_isa_t isa, data_type_t kernel_dt>
status_t jit_uni_dw_conv_bwd_weights_kernel<isa, kernel_dt>::init_conf(
        jit_conv_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &diff_weights_d,
        const memory_desc_wrapper &diff_dst_d, int nthreads) {

    jcp.dwei_dt = cd.diff_weights_desc.data_type;
    const bool is_bf16 = src_d.data_type() == data_type::bf16;

    if (!mayiuse(isa) || (is_bf16 && !mayiuse(avx512_core)))
        return status::unimplemented;

    jcp.ngroups = diff_weights_d.dims()[0];
    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;

    jcp.is_depthwise = true && with_groups && everyone_is(1, jcp.oc, jcp.ic);

    if (!jcp.is_depthwise)
        return status::unimplemented;

    jcp.ch_block = one_of(isa, avx512_common, avx512_core) ? 16 : 8;

    jcp.mb = src_d.dims()[0];

    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = diff_dst_d.dims()[2];
    jcp.ow = diff_dst_d.dims()[3];

    jcp.kh = diff_weights_d.dims()[3];
    jcp.kw = diff_weights_d.dims()[4];

    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];

    jcp.t_pad = cd.padding[0][0];
    jcp.b_pad = cd.padding[1][0];

    jcp.l_pad = cd.padding[0][1];
    jcp.r_pad = cd.padding[1][1];

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;

    jcp.src_fmt = src_d.format();

    jcp.with_bias = cd.diff_bias_desc.format != memory_format::undef;

    auto desired_act_fmt
            = one_of(isa, avx512_common, avx512_core) ? nChw16c : nChw8c;
    auto desired_wei_fmt
            = one_of(isa, avx512_common, avx512_core) ? Goihw16g : Goihw8g;

    bool args_ok = true && src_d.format() == desired_act_fmt
            && diff_weights_d.format() == desired_wei_fmt
            && diff_dst_d.format() == desired_act_fmt
            && one_of(cd.bias_desc.format, memory_format::undef, any, x)
            && jcp.ngroups % jcp.ch_block == 0 && jcp.dilate_h == 0
            && jcp.dilate_w == 0 && jcp.kw <= 3
            && jcp.oh == (jcp.ihp - jcp.kh) / jcp.stride_h + 1
            && jcp.ow == (jcp.iwp - jcp.kw) / jcp.stride_w + 1;
    if (!args_ok)
        return status::unimplemented;

    if (!IMPLICATION(is_bf16, desired_act_fmt == mkldnn_nChw16c
                        && desired_wei_fmt == mkldnn_Goihw16g))
        return status::unimplemented;

    jcp.nb_ch = jcp.ngroups / jcp.ch_block;

    /* kernel applicability check wrt boundaries
     * the conditions are quite general across the kernels we have,
     * but ideally the check should belong to a specific kernel... */
    const int max_hpad = (jcp.kh - 1 + 1) / 2;
    const int max_wpad = (jcp.kw - 1 + 1) / 2;
    const bool boundaries_ok = true && jcp.t_pad <= max_hpad
            && jcp.b_pad <= max_hpad && jcp.l_pad <= max_wpad
            && jcp.r_pad <= max_wpad;
    if (!boundaries_ok)
        return status::unimplemented;

    jcp.is_cpx = (mayiuse(avx512_core_bf16)) ? true : false;

    /* BF16: accumulation of output happens in f32, down-conversion to bf16
     * happens during the reduction phase. */
    jcp.typesize_out = sizeof(float);
    jcp.typesize_in = src_d.data_type() == data_type::bf16
            ? sizeof(mkldnn_bfloat16_t)
            : sizeof(float);

    balance(jcp, nthreads);

    return status::success;
}

template <cpu_isa_t isa, data_type_t kernel_dt>
void jit_uni_dw_conv_bwd_weights_kernel<isa, kernel_dt>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    /* Notes: if splitting thread work on 'mb', then a reduction has to take
     * place. Hence, book a per-thread, local weights-buffer for the
     * reduction */
    if (jcp.nthr_mb > 1) {
        const size_t mb = jcp.dwei_dt == data_type::bf16 ? jcp.nthr_mb
                                                           : jcp.nthr_mb - 1;
        const size_t wei_size = jcp.ngroups * jcp.kh * jcp.kw;
        scratchpad.book(key_conv_wei_reduction, sizeof(float) * wei_size * mb);

        if (jcp.with_bias)
            scratchpad.book(key_conv_bia_reduction,
                    sizeof(float) * jcp.ngroups * (jcp.nthr_mb - 1));
    } else if (jcp.nthr_mb == 1 && jcp.dwei_dt == data_type::bf16) {
        const size_t wei_size = jcp.ngroups * jcp.kh * jcp.kw;
        scratchpad.book(key_conv_wei_reduction, sizeof(float) * wei_size);
    }
}

template <cpu_isa_t isa, data_type_t kernel_dt>
void jit_uni_dw_conv_bwd_weights_kernel<isa, kernel_dt>::balance(
        jit_conv_conf_t &jcp, int nthreads) {
    jcp.nthr = nthreads;
    jcp.nthr_g = jcp.nthr_mb = 1;

    /* Basic-Heuristics for parallel strategy:
     * 1) Tries to parallel on the number of Groups (g) where tasks are
     * independent. Otherwise,
     * 2) Tries to split the work across g and MiniBatch (mb).
     * Parallelizing on mb requires computing a reduction for weights.
     *
     * NOTE: because of 'task partitioning' scheme, there will be unbalanced
     * per-thread load when the number of threads is high (e.g. > 16).
     */
    jcp.nthr_g = nstl::min(jcp.nb_ch, jcp.nthr);
    jcp.nthr_mb = nstl::min(nstl::max(1, jcp.nthr / jcp.nthr_g), jcp.mb);

    jcp.nthr = jcp.nthr_g * jcp.nthr_mb;
}

template struct jit_uni_dw_conv_bwd_weights_kernel<avx512_core,
        data_type::bf16>;
template struct jit_uni_dw_conv_bwd_weights_kernel<avx512_common,
        data_type::f32>;
template struct jit_uni_dw_conv_bwd_weights_kernel<avx2, data_type::f32>;
template struct jit_uni_dw_conv_bwd_weights_kernel<sse42, data_type::f32>;
}
}
}
#endif /* JIT_UNI_DW_CONVOLUTION_UTILS_HPP */
