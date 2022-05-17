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

#ifndef CPU_X64_JIT_UNI_FORK_DW_CONV_KERNEL_UTILS_HPP
#define CPU_X64_JIT_UNI_FORK_DW_CONV_KERNEL_UTILS_HPP

#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"

#include "cpu/x64/jit_avx512_core_fork_bf16_dw_conv_kernel.hpp"
#include "cpu/x64/jit_uni_fork_dw_conv_kernel_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, data_type_t kernel_dt>
struct jit_uni_fork_dw_conv_fwd_kernel {

    jit_uni_fork_dw_conv_fwd_kernel(const jit_conv_conf_t &ajcp, const memory_desc_t &dst_md, const primitive_attr_t &attr) : ker_(nullptr) {
        ker_ = new jit_kernel_t(ajcp, dst_md, attr);
    }

    status_t create_kernel() { return ker_->create_kernel(); }
    ~jit_uni_fork_dw_conv_fwd_kernel() { delete ker_; }

    static bool post_ops_ok(jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_md,
            memory_desc_t &weights_md, memory_desc_t &bias_md,
            memory_desc_t &dst_md, const primitive_attr_t &attr);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    jit_generator *ker() const { return ker_; }
    void operator()(const jit_conv_call_s *p) const { (*ker_)(p); }

private:
    using jit_kernel_t = typename utils::conditional<isa == avx512_core
                    && kernel_dt == data_type::bf16,
            jit_avx512_fork_dw_conv_fwd_kernel_bf16,
            jit_uni_fork_dw_conv_fwd_kernel_f32<isa>>::type;
    jit_kernel_t *ker_;
};

template <cpu_isa_t isa, data_type_t kernel_dt>
bool jit_uni_fork_dw_conv_fwd_kernel<isa, kernel_dt>::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;
    auto all_post_ops_supported = [&]() {
        bool ok = true;

        for (int i = 0; i < p.len(); i++) {
            ok = ok && utils::one_of(p.entry_[i].kind, primitive_kind::sum, primitive_kind::eltwise, primitive_kind::depthwise, primitive_kind::quantization);
        }
        return ok;
    };
    auto contain = [&](dnnl::impl::primitive_kind_t kind) { return p.find(kind) != -1; };
    auto position = [&](dnnl::impl::primitive_kind_t kind) { return p.find(kind); };
    auto count = [&](dnnl::impl::primitive_kind_t kind) { return p.count(kind); };

    return all_post_ops_supported() &&
           count(primitive_kind::sum) <= 1 &&
           IMPLICATION(contain(primitive_kind::sum), position(primitive_kind::sum) == 0);
}

template <cpu_isa_t isa, data_type_t kernel_dt>
status_t jit_uni_fork_dw_conv_fwd_kernel<isa, kernel_dt>::init_conf(
        jit_conv_conf_t &jcp, const convolution_desc_t &cd,
        memory_desc_t &src_md, memory_desc_t &weights_md,
        memory_desc_t &bias_md, memory_desc_t &dst_md,
        const primitive_attr_t &attr) {

    using namespace dnnl::impl::format_tag;
    using namespace dnnl::impl::utils;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const int ndims = src_d.ndims();
    const auto blocked_tag = one_of(isa, avx512_common, avx512_core) ?
                             pick(ndims - 3, nCw16c, nChw16c, nCdhw16c) :
                             pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
    const auto wei_tag = one_of(isa, avx512_common, avx512_core) ?
                         pick(ndims - 3, Goiw16g, Goihw16g, Goidhw16g) :
                         pick(ndims - 3, Goiw8g, Goihw8g, Goidhw8g);
    const auto nxc_tag = pick(ndims - 3, nwc, nhwc, ndhwc);
    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    if (src_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(src_md, blocked_tag));
        jcp.src_tag = blocked_tag;
    } else {
        jcp.src_tag = src_d.mb_stride_relaxed_match(blocked_tag, nxc_tag);
    }

    if (weights_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(weights_md, wei_tag));
        jcp.wei_tag = wei_tag;
    } else {
        jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);
    }

    if (dst_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(dst_md, blocked_tag));
        jcp.dst_tag = blocked_tag;
    } else {
        jcp.dst_tag = dst_d.mb_stride_relaxed_match(blocked_tag, nxc_tag);
    }

    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, format_tag::x));
    }

    if (jcp.dst_tag != jcp.src_tag) return status::unimplemented;
    const auto data_tag = jcp.src_tag;
    const bool is_data_layout_nxc = data_tag == nxc_tag;

    const bool is_bf16 = src_d.data_type() == data_type::bf16;
    // 3D bf16 fork DW kernel does not support 3D convolution
    if (is_bf16 && ndims == 5) return status::unimplemented;

    jcp.dst_dt = cd.dst_desc.data_type;
    jcp.isa = (is_bf16 && mayiuse(avx512_core_bf16)) ? avx512_core_bf16 : isa;

    if (!mayiuse(isa) || (is_bf16 && !mayiuse(avx512_core)))
        return status::unimplemented;

    const int simd_w = one_of(isa, avx512_common, avx512_core) ? 16 : 8;

    jcp.prop_kind = cd.prop_kind;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    if (!with_groups) return status::unimplemented;

    jcp.ndims = ndims;

    jcp.ngroups = weights_d.dims()[0];
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1];
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1];

    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];

    jcp.kd = (ndims == 5) ? weights_d.dims()[3] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[ndims - 1];
    jcp.kw = weights_d.dims()[ndims];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.back_pad = (ndims == 5) ? cd.padding[1][0] : 0;
    jcp.b_pad = (ndims == 3) ? 0 : cd.padding[1][ndims - 4];
    jcp.r_pad = cd.padding[1][ndims - 3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    jcp.loop_order = loop_ngcw;

    if (is_data_layout_nxc) {
        jcp.loop_order = loop_nhwcg;
    }

    if (!post_ops_ok(jcp, attr))
            return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise)
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;

    jcp.post_ops = p;

    bool ok_to_pad_channels = true
        && !is_data_layout_nxc
        && jcp.oc == jcp.ngroups
        && jcp.ic == jcp.ngroups
        && one_of(isa, avx512_common, avx512_core, avx2);
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.oc, simd_w);
        jcp.ngroups = rnd_up(jcp.ngroups, simd_w);
    }

    bool args_ok = true && jcp.oc == jcp.ngroups && jcp.ic == jcp.ngroups
                   && IMPLICATION(!is_data_layout_nxc, jcp.ngroups % simd_w == 0)
                   && jcp.wei_tag == wei_tag
                   && data_tag != format_tag::undef && jcp.ic <= src_d.padded_dims()[1]
                   && jcp.oc <= dst_d.padded_dims()[1]
                   && jcp.ngroups <= weights_d.padded_dims()[0];
    if (!args_ok) return status::unimplemented;

    jcp.typesize_out = jcp.dst_dt == data_type::bf16 ? sizeof(bfloat16_t)
                                                     : sizeof(float);
    jcp.typesize_in = src_d.data_type() == data_type::bf16
            ? sizeof(bfloat16_t)
            : sizeof(float);

    jcp.ur_w = is_bf16 ? (isa_has_bf16(jcp.isa) ? 6 : 4)
                       : isa == avx512_common ? 6 : isa == avx2 ? 4 : 3;

    jcp.ch_block = simd_w;
    jcp.nb_ch = div_up(jcp.oc, jcp.ch_block);
    jcp.nb_ch_blocking
            = one_of(isa, avx512_common, avx512_core) ? 4 : isa == avx2 ? 3 : 2;
    if (jcp.nb_ch < jcp.nb_ch_blocking)
        jcp.nb_ch_blocking = jcp.nb_ch;

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;

    return status::success;
}

template <cpu_isa_t isa, data_type_t kernel_dt>
void jit_uni_fork_dw_conv_fwd_kernel<isa, kernel_dt>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    using namespace dnnl::impl::memory_tracking::names;
    if (jcp.bia_dt == data_type::bf16)
        scratchpad.book<float>(key_conv_bias_bf16_convert_wsp, jcp.oc);
    else if (jcp.with_bias && jcp.oc_without_padding != jcp.oc)
        scratchpad.book<float>(key_conv_padded_bias, jcp.oc);
}

template struct jit_uni_fork_dw_conv_fwd_kernel<avx512_core, data_type::bf16>;
template struct jit_uni_fork_dw_conv_fwd_kernel<avx512_common, data_type::f32>;
template struct jit_uni_fork_dw_conv_fwd_kernel<avx2, data_type::f32>;
template struct jit_uni_fork_dw_conv_fwd_kernel<sse41, data_type::f32>;

template <cpu_isa_t isa, data_type_t kernel_dt>
struct jit_uni_fork_dw_conv_bwd_data_kernel {

    jit_uni_fork_dw_conv_bwd_data_kernel(const jit_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : ker_(nullptr) {
        ker_ = new jit_kernel_t(ajcp, attr);
    }

    status_t create_kernel() { return ker_->create_kernel(); }
    ~jit_uni_fork_dw_conv_bwd_data_kernel() { delete ker_; }

    static bool post_ops_ok(const primitive_attr_t &attr);

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &diff_src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &diff_dst_d, const primitive_attr_t &attr);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    void operator()(const jit_conv_call_s *p) const { (*ker_)(p); }

private:
    using jit_kernel_t = typename utils::conditional<isa == avx512_core
                    && kernel_dt == data_type::bf16,
            jit_avx512_fork_dw_conv_bwd_data_kernel_bf16,
            jit_uni_fork_dw_conv_bwd_data_kernel_f32<isa>>::type;
    jit_kernel_t *ker_;

    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_uni_fork_dw_conv_bwd_data_kernel);
};

template <cpu_isa_t isa, data_type_t kernel_dt>
bool jit_uni_fork_dw_conv_bwd_data_kernel<isa, kernel_dt>::post_ops_ok(const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;
    if (p.len() > 1)
        return false;

    auto all_post_ops_supported = [&]() {
        bool ok = true;

        for (int i = 0; i < p.len(); i++) {
            ok = ok && utils::one_of(p.entry_[i].kind, primitive_kind::depthwise);
        }
        return ok;
    };

    return all_post_ops_supported();
}

template <cpu_isa_t isa, data_type_t kernel_dt>
status_t jit_uni_fork_dw_conv_bwd_data_kernel<isa, kernel_dt>::init_conf(
        jit_conv_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d, const primitive_attr_t &attr) {
    using namespace dnnl::impl::format_tag;
    using namespace dnnl::impl::utils;

    jcp.dsrc_dt = cd.diff_src_desc.data_type;
    const bool is_bf16 = diff_dst_d.data_type() == data_type::bf16;
    jcp.isa = (is_bf16 && mayiuse(avx512_core_bf16)) ? avx512_core_bf16 : isa;

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

    if (!post_ops_ok(attr))
        return status::unimplemented;

    jcp.post_ops = attr.post_ops_;

    bool ok_to_pad_channels = true && jcp.oc == jcp.ngroups
            && jcp.ic == jcp.ngroups
            && one_of(isa, avx512_common, avx512_core, avx2);
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.oc, simd_w);
        jcp.ngroups = rnd_up(jcp.ngroups, simd_w);
    }

    auto dat_tag = one_of(isa, avx512_common, avx512_core) ? nChw16c : nChw8c;
    auto wei_tag = one_of(isa, avx512_common, avx512_core) ? Goihw16g : Goihw8g;

    jcp.src_tag = diff_src_d.mb_stride_relaxed_match(dat_tag);
    jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);
    jcp.dst_tag = diff_dst_d.mb_stride_relaxed_match(dat_tag);

    bool args_ok = true && jcp.oc == jcp.ngroups && jcp.ic == jcp.ngroups
            && jcp.ngroups % simd_w == 0 && jcp.dilate_h == 0
            && jcp.dilate_w == 0 && jcp.src_tag == dat_tag
            && jcp.wei_tag == wei_tag && jcp.dst_tag == dat_tag
            && jcp.oh == (jcp.ihp - jcp.kh) / jcp.stride_h + 1
            && jcp.ow == (jcp.iwp - jcp.kw) / jcp.stride_w + 1
            && jcp.ic <= diff_src_d.padded_dims()[1]
            && jcp.oc <= diff_dst_d.padded_dims()[1]
            && jcp.ngroups <= weights_d.padded_dims()[0];
    if (!args_ok) return status::unimplemented;

    jcp.typesize_out = types::data_type_size(diff_src_d.data_type());
    jcp.typesize_in = types::data_type_size(diff_dst_d.data_type());

    jcp.ur_w = is_bf16 ? (isa_has_bf16(jcp.isa) ? 6 : 4)
                       : isa == avx512_common ? 6 : isa == avx2 ? 4 : 3;

    jcp.ch_block = simd_w;
    jcp.nb_ch = jcp.ic / jcp.ch_block;
    jcp.nb_ch_blocking
            = one_of(isa, avx512_common, avx512_core) ? 4 : isa == avx2 ? 3 : 2;
    if (jcp.nb_ch < jcp.nb_ch_blocking) jcp.nb_ch_blocking = jcp.nb_ch;

    return status::success;
}

template <cpu_isa_t isa, data_type_t kernel_dt>
void jit_uni_fork_dw_conv_bwd_data_kernel<isa, kernel_dt>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    UNUSED(scratchpad);
    UNUSED(jcp);
}

template struct jit_uni_fork_dw_conv_bwd_data_kernel<avx512_core, data_type::bf16>;
template struct jit_uni_fork_dw_conv_bwd_data_kernel<avx512_common, data_type::f32>;
template struct jit_uni_fork_dw_conv_bwd_data_kernel<avx2, data_type::f32>;
template struct jit_uni_fork_dw_conv_bwd_data_kernel<sse41, data_type::f32>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif /* CPU_X64_JIT_uni_fork_dw_CONV_KERNEL_UTILS_HPP */
