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

#include "cpu/x64/jit_uni_dw_conv_kernel_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace data_type;

template <cpu_isa_t isa, data_type_t kernel_dt>
status_t jit_uni_dw_conv_fwd_kernel<isa, kernel_dt>::init_conf(
        jit_conv_conf_t &jcp, const convolution_desc_t &cd,
        memory_desc_t &src_md, memory_desc_t &weights_md,
        memory_desc_t &bias_md, memory_desc_t &dst_md, primitive_attr_t &attr) {

    using namespace dnnl::impl::format_tag;
    using namespace dnnl::impl::utils;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const int ndims = src_d.ndims();
    // Currently this kernel only supports 2D convolutions.
    if (ndims != 4) return status::unimplemented;

    jcp.prop_kind = cd.prop_kind;

    const auto blocked_tag
            = one_of(isa, avx512_common, avx512_core) ? nChw16c : nChw8c;
    const auto wei_tag
            = one_of(isa, avx512_common, avx512_core) ? Goihw16g : Goihw8g;
    const auto nxc_tag = nhwc;
    const auto def_tag
            = (mayiuse(avx512_core)
                      && jcp.prop_kind == prop_kind::forward_inference)
            ? nxc_tag
            : blocked_tag;

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    if (src_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(src_md, def_tag));
        jcp.src_tag = def_tag;
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
        CHECK(memory_desc_init_by_tag(dst_md, def_tag));
        jcp.dst_tag = def_tag;
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

    jcp.dst_dt = cd.dst_desc.data_type;
    jcp.isa = (is_bf16 && mayiuse(avx512_core_bf16)) ? avx512_core_bf16 : isa;

    if (!mayiuse(isa) || (is_bf16 && !mayiuse(avx512_core)))
        return status::unimplemented;

    const int simd_w = one_of(isa, avx512_common, avx512_core) ? 16 : 8;

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

    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];

    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_in = types::data_type_size(src_d.data_type());

    jcp.loop_order = loop_ngcw;

    jcp.ur_w = is_bf16 ? (isa_has_bf16(jcp.isa) ? 6 : 4)
                       : isa == avx512_common ? 6 : isa == avx2 ? 4 : 3;
    jcp.ur_w = nstl::min(jcp.ur_w, jcp.ow);

    jcp.ch_block = simd_w;
    jcp.nb_ch = div_up(jcp.oc, jcp.ch_block);
    jcp.nb_ch_blocking
            = one_of(isa, avx512_common, avx512_core) ? 4 : isa == avx2 ? 3 : 2;
    if (jcp.nb_ch < jcp.nb_ch_blocking) jcp.nb_ch_blocking = jcp.nb_ch;

    if (is_data_layout_nxc) {
        jcp.loop_order = loop_nhwcg;
        const int resrc_depthwise_ur_w = (31 - jcp.kw + jcp.stride_w)
                / (jcp.nb_ch_blocking + jcp.stride_w);
        jcp.is_resrc_depthwise = (!is_bf16)
                && one_of(isa, avx512_common, avx512_core)
                && jcp.stride_w < jcp.kw && jcp.kw <= 5 && jcp.dilate_w == 0
                && resrc_depthwise_ur_w >= 2;
        if (jcp.is_resrc_depthwise) {
            jcp.ur_w = nstl::min(jcp.ow, resrc_depthwise_ur_w);
        }
        bool cache_aliasing
                = (jcp.ngroups * jcp.iw * jcp.typesize_in) % 1024 == 0;
        if (cache_aliasing) {
            // currently only tuned for mobilenet-v1 shapes
            const int limit = jcp.ow > 7 ? 7 : 4;
            jcp.ur_w = nstl::min(jcp.ur_w, limit);
        }
    }

    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh);
    bool kernel_outside_src = false || ext_kw <= jcp.l_pad
            || ext_kw <= jcp.r_pad || ext_kh <= jcp.t_pad
            || ext_kh <= jcp.b_pad;
    if (kernel_outside_src) return status::unimplemented;
    int r_pad_no_tail = nstl::max(0,
            calculate_end_padding(jcp.l_pad, jcp.ow - jcp.ur_w_tail, jcp.iw,
                    jcp.stride_w, ext_kw));
    if (jcp.l_pad > jcp.ur_w || r_pad_no_tail > jcp.ur_w)
        return status::unimplemented;

    CHECK(attr.set_default_formats(&dst_md));

    const auto &post_ops = attr.post_ops_;

    jcp.with_sum = post_ops.find(primitive_kind::sum) != -1;
    const int eltwise_ind = post_ops.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) jcp.eltwise = post_ops.entry_[eltwise_ind].eltwise;
    const int binary_ind = post_ops.find(primitive_kind::binary);
    jcp.with_binary = binary_ind != -1;
    if (jcp.with_binary) {
        using namespace dnnl::impl::cpu::binary_injector_utils;
        std::tie(jcp.with_binary_per_oc_bcast, jcp.with_binary_no_bcast)
                = bcast_strategies_present_tup(post_ops.entry_, dst_d,
                        broadcasting_strategy_t::per_oc,
                        broadcasting_strategy_t::no_broadcast);
    }
    jcp.with_depthwise = post_ops.find(primitive_kind::depthwise) != -1;
    jcp.with_quantization = post_ops.find(primitive_kind::quantization) != -1;

    jcp.post_ops = post_ops;

    using namespace injector;
    static constexpr bool sum_at_pos_0_only = true;
    static constexpr bool sum_requires_scale_one = true;
    const bool post_ops_ok_ = post_ops_ok({isa, {eltwise, binary, sum, depthwise, quantization},
            jcp.post_ops, &dst_d, sum_at_pos_0_only, sum_requires_scale_one});
    if (!post_ops_ok_) return status::unimplemented;

    const bool ok_to_pad_channels = true && !is_data_layout_nxc
            && jcp.oc == jcp.ngroups && jcp.ic == jcp.ngroups
            && one_of(isa, avx512_common, avx512_core, avx2, sse41);
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.oc, simd_w);
        jcp.ngroups = rnd_up(jcp.ngroups, simd_w);
    }

    const bool args_ok = true && jcp.oc == jcp.ngroups && jcp.ic == jcp.ngroups
            && IMPLICATION(!is_data_layout_nxc, jcp.ngroups % simd_w == 0)
            && jcp.wei_tag == wei_tag && data_tag != format_tag::undef
            && jcp.ic <= src_d.padded_dims()[1]
            && jcp.oc <= dst_d.padded_dims()[1]
            && jcp.ngroups <= weights_d.padded_dims()[0];
    if (!args_ok) return status::unimplemented;

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;

    return status::success;
}

template <cpu_isa_t isa, data_type_t kernel_dt>
void jit_uni_dw_conv_fwd_kernel<isa, kernel_dt>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    using namespace dnnl::impl::memory_tracking::names;
    if (jcp.bia_dt == data_type::bf16)
        scratchpad.book<float>(key_conv_bias_bf16_convert_wsp, jcp.oc);
    else if (jcp.with_bias && jcp.oc_without_padding != jcp.oc)
        scratchpad.book<float>(key_conv_padded_bias, jcp.oc);
}

template <cpu_isa_t isa, data_type_t kernel_dt>
bool jit_uni_dw_conv_bwd_data_kernel<isa, kernel_dt>::post_ops_ok(const primitive_attr_t &attr) {
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
status_t jit_uni_dw_conv_bwd_data_kernel<isa, kernel_dt>::init_conf(
        jit_conv_conf_t &jcp, const convolution_desc_t &cd,
        memory_desc_t &diff_src_md, memory_desc_t &weights_md,
        memory_desc_t &diff_dst_md, const primitive_attr_t &attr) {
    using namespace dnnl::impl::format_tag;
    using namespace dnnl::impl::utils;

    const memory_desc_wrapper diff_src_d(&diff_src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper diff_dst_d(&diff_dst_md);

    jcp.dsrc_dt = cd.diff_src_desc.data_type;
    const bool is_bf16 = diff_dst_d.data_type() == bf16;
    jcp.isa = (is_bf16 && mayiuse(avx512_core_bf16)) ? avx512_core_bf16 : isa;

    if (!mayiuse(isa) || (is_bf16 && !mayiuse(avx512_core)))
        return status::unimplemented;

    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;
    if (!with_groups) return status::unimplemented;

    const int ndims = diff_src_d.ndims();
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

    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];

    const int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    const int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh);

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;

    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_blocked
            = one_of(isa, avx512_common, avx512_core) ? nChw16c : nChw8c;
    const auto wei_tag
            = one_of(isa, avx512_common, avx512_core) ? Goihw16g : Goihw8g;

    auto curr_src_tag
            = diff_src_d.mb_stride_relaxed_match(dat_tag_nxc, dat_tag_blocked);
    auto curr_dst_tag
            = diff_dst_d.mb_stride_relaxed_match(dat_tag_nxc, dat_tag_blocked);
    bool is_data_layout_nxc
            = utils::everyone_is(dat_tag_nxc, curr_src_tag, curr_dst_tag);
    auto dat_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_blocked;

    if (diff_src_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_src_md, dat_tag_blocked));
        jcp.src_tag = dat_tag_blocked;
    } else if (curr_src_tag != dat_tag)
        return status::unimplemented;
    else
        jcp.src_tag = dat_tag;

    if (diff_dst_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_dst_md, dat_tag_blocked));
        jcp.dst_tag = dat_tag_blocked;
    } else if (curr_dst_tag != dat_tag)
        return status::unimplemented;
    else
        jcp.dst_tag = dat_tag;

    if (weights_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(weights_md, wei_tag));
        jcp.wei_tag = wei_tag;
    } else {
        jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);
    }

    // No support for mixed types between SRC and DIFF_DST tensors
    if (!everyone_is(dat_tag, jcp.src_tag, jcp.dst_tag)
            || jcp.wei_tag != wei_tag)
        return status::unimplemented;

    // note: sse41 uses 'ch_block = 8' where the value is derived
    // from: 'simd_w_ * reg_repeats_ = 4 * 2'
    jcp.ch_block = one_of(isa, avx512_common, avx512_core) ? 16 : 8;

    if (!post_ops_ok(attr))
        return status::unimplemented;

    jcp.post_ops = attr.post_ops_;

    bool ok_to_pad_channels = !is_data_layout_nxc && jcp.oc == jcp.ngroups
            && jcp.ic == jcp.ngroups
            && one_of(isa, avx512_common, avx512_core, avx2);
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.ch_block);
        jcp.ic = rnd_up(jcp.oc, jcp.ch_block);
        jcp.ngroups = rnd_up(jcp.ngroups, jcp.ch_block);
    }

    bool args_ok = true && jcp.oc == jcp.ngroups && jcp.ic == jcp.ngroups
            && IMPLICATION(!is_data_layout_nxc, jcp.ngroups % jcp.ch_block == 0)
            && jcp.dilate_h == 0 && jcp.dilate_w == 0
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

    jcp.loop_order = is_data_layout_nxc ? loop_nhwcg : loop_ngcw;

    jcp.ch_tail = jcp.ngroups % jcp.ch_block;
    jcp.nb_ch = div_up(jcp.ic, jcp.ch_block);
    jcp.nb_ch_blocking
            = one_of(isa, avx512_common, avx512_core) ? 4 : isa == avx2 ? 3 : 2;
    if (jcp.nb_ch < jcp.nb_ch_blocking) jcp.nb_ch_blocking = jcp.nb_ch;

    return status::success;
}

template <cpu_isa_t isa, data_type_t kernel_dt>
void jit_uni_dw_conv_bwd_data_kernel<isa, kernel_dt>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    UNUSED(scratchpad);
    UNUSED(jcp);
}

template <cpu_isa_t isa, data_type_t kernel_dt>
status_t jit_uni_dw_conv_bwd_weights_kernel<isa, kernel_dt>::init_conf(
        jit_conv_conf_t &jcp, const convolution_desc_t &cd,
        memory_desc_t &src_md, memory_desc_t &diff_weights_md,
        memory_desc_t &diff_bias_md, memory_desc_t &diff_dst_md, int nthreads) {
    using namespace dnnl::impl::format_tag;
    using namespace dnnl::impl::utils;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper diff_weights_d(&diff_weights_md);
    const memory_desc_wrapper diff_bias_d(&diff_bias_md);
    const memory_desc_wrapper diff_dst_d(&diff_dst_md);

    jcp.dwei_dt = cd.diff_weights_desc.data_type;
    const int ndims = src_d.ndims();
    const bool is_bf16 = src_d.data_type() == data_type::bf16;
    jcp.isa = (is_bf16 && mayiuse(avx512_core_bf16)) ? avx512_core_bf16 : isa;

    if (!mayiuse(isa) || (is_bf16 && !mayiuse(avx512_core)))
        return status::unimplemented;

    jcp.ngroups = diff_weights_d.dims()[0];
    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = diff_dst_d.dims()[1];
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;

    jcp.is_depthwise = true && with_groups && everyone_is(1, jcp.oc, jcp.ic);

    if (!jcp.is_depthwise) return status::unimplemented;

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
    jcp.l_pad = cd.padding[0][1];

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];

    jcp.with_bias = cd.diff_bias_desc.format_kind != format_kind::undef;

    const int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    const int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    jcp.r_pad = nstl::max(0,
            calculate_end_padding(
                    jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw));
    jcp.b_pad = nstl::max(0,
            calculate_end_padding(
                    jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh));

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;

    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_blocked = one_of(isa, avx512_common, avx512_core)
            ? nChw16c
            : nChw8c; // dnnl_aBcd16b
    const auto wei_tag
            = one_of(isa, avx512_common, avx512_core) ? Goihw16g : Goihw8g;
    auto curr_src_tag = src_d.matches_one_of_tag(dat_tag_nxc, dat_tag_blocked);
    auto curr_dst_tag
            = diff_dst_d.matches_one_of_tag(dat_tag_nxc, dat_tag_blocked);

    bool is_data_layout_nxc
            = utils::everyone_is(dat_tag_nxc, curr_src_tag, curr_dst_tag);

    auto dat_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_blocked;

    if (src_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(src_md, dat_tag_blocked));
        jcp.src_tag = dat_tag_blocked;
    } else if (curr_src_tag != dat_tag)
        return status::unimplemented;
    else
        jcp.src_tag = dat_tag;

    if (diff_dst_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_dst_md, dat_tag_blocked));
        jcp.dst_tag = dat_tag_blocked;
    } else if (curr_dst_tag != dat_tag)
        return status::unimplemented;
    else
        jcp.dst_tag = dat_tag;

    if (diff_weights_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_weights_md, wei_tag));
        jcp.wei_tag = wei_tag;
    } else {
        jcp.wei_tag = diff_weights_d.matches_one_of_tag(wei_tag);
    }

    // No support for mixed types between SRC and DIFF_DST tensors
    if (!everyone_is(dat_tag, jcp.src_tag, jcp.dst_tag)
            || jcp.wei_tag != wei_tag)
        return status::unimplemented;

    if (jcp.with_bias) {
        if (diff_bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_bias_md, x));
    }

    jcp.ch_block = one_of(isa, avx512_common, avx512_core) ? 16 : 8;
    jcp.ch_tail = jcp.oc_without_padding % jcp.ch_block;

    // note: bf16 to be supported in the next commit
    bool ok_to_pad_channels = !is_data_layout_nxc
            && one_of(isa, avx512_common, avx512_core, avx2);
    if (ok_to_pad_channels) { jcp.ngroups = rnd_up(jcp.ngroups, jcp.ch_block); }

    bool args_ok = true
            && IMPLICATION(!is_data_layout_nxc, jcp.ngroups % jcp.ch_block == 0)
            && jcp.dilate_h == 0 && jcp.dilate_w == 0 && jcp.kw <= 3
            && jcp.stride_w <= jcp.kw // no gaps in kernel
            && jcp.oh == (jcp.ihp - jcp.kh) / jcp.stride_h + 1
            && jcp.ow == (jcp.iwp - jcp.kw) / jcp.stride_w + 1;
    if (!args_ok) return status::unimplemented;

    jcp.nb_ch = div_up(jcp.ngroups, jcp.ch_block);

    // Note: avx2 can't do masked_fma and would require extra Vmms
    // for byte_load.
    // TODO: enable 'is_fast_depthwise' for bf16 if it offers performance
    // improvement.
    jcp.is_fast_depthwise = !is_bf16 && is_data_layout_nxc
            && one_of(isa, avx512_common, avx2);
    constexpr int max_registers = isa == avx512_common ? 32 : 16;
    // Note: anything larger than 4 didn't show significant speedup
    const int max_isa_unroll = jcp.is_fast_depthwise ? 4 : 1;
    int max_ch_unroll = nstl::min(max_isa_unroll, max_registers / (2 * jcp.kw));
    jcp.nb_ch_blocking = nstl::min(jcp.nb_ch, max_ch_unroll);

    /* kernel applicability check wrt boundaries
     * the conditions are quite general across the kernels we have,
     * but ideally the check should belong to a specific kernel... */
    const int max_hpad = (jcp.kh - 1 + 1) / 2;
    const int max_wpad = (jcp.kw - 1 + 1) / 2;
    const int min_ih = jcp.kh + nstl::modulo(-jcp.t_pad, jcp.stride_h);
    const bool boundaries_ok = true && jcp.t_pad <= max_hpad
            && jcp.b_pad <= max_hpad && jcp.l_pad <= max_wpad
            && jcp.r_pad <= max_wpad
            // input must fully accommodate the filter
            && jcp.ih >= min_ih
            // non-unit padding must be a multiple of the stride
            && IMPLICATION(jcp.t_pad > 1, jcp.t_pad % jcp.stride_h == 0)
            && IMPLICATION(jcp.b_pad > 1, jcp.b_pad % jcp.stride_h == 0);
    if (!boundaries_ok) return status::unimplemented;

    /* BF16: accumulation of output happens in f32, down-conversion to bf16
     * happens during the reduction phase. */
    jcp.typesize_out = sizeof(float);
    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.bia_dt = jcp.with_bias ? cd.diff_bias_desc.data_type : data_type::undef;

    jcp.harness = is_data_layout_nxc ? harness_nxc : harness_mb_reduction;

    balance(jcp, nthreads);

    return status::success;
}

template <cpu_isa_t isa, data_type_t kernel_dt>
void jit_uni_dw_conv_bwd_weights_kernel<isa, kernel_dt>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    using namespace dnnl::impl::memory_tracking::names;

    if (jcp.harness == harness_mb_reduction) {
        /* Notes: if splitting thread work on 'mb', then a reduction has to take
         * place. Hence, book a per-thread, local weights-buffer for the
         * reduction */
        if (jcp.nthr_mb > 1) {
            const size_t mb = jcp.dwei_dt == data_type::bf16 ? jcp.nthr_mb
                                                             : jcp.nthr_mb - 1;
            const size_t wei_size = jcp.ngroups * jcp.kh * jcp.kw;
            scratchpad.book<float>(key_conv_wei_reduction, wei_size * mb);

            if (jcp.with_bias)
                scratchpad.book<float>(key_conv_bia_reduction,
                        jcp.ngroups * (jcp.nthr_mb - 1));
        } else if (jcp.nthr_mb == 1 && jcp.dwei_dt == data_type::bf16) {
            const size_t wei_size = jcp.ngroups * jcp.kh * jcp.kw;
            scratchpad.book<float>(key_conv_wei_reduction, wei_size);
        }
    } else if (jcp.harness == harness_nxc) {
        if (jcp.nthr > 1 || jcp.dwei_dt == data_type::bf16) {
            assert(jcp.nthr > 0); // redundant check
            const size_t buff_count
                    = jcp.dwei_dt == data_type::bf16 ? jcp.nthr : jcp.nthr - 1;

            // note: because of weights blocked format, buffer is padded
            // across ch_block
            const size_t wei_size = utils::rnd_up(jcp.ngroups, jcp.ch_block)
                    * jcp.kh * jcp.kw;
            scratchpad.book<float>(
                    key_conv_wei_reduction, wei_size * buff_count);

            if (jcp.with_bias) {
                scratchpad.book<float>(
                        key_conv_bia_reduction, jcp.ngroups * buff_count);
            }
        }
    }

    if (jcp.bia_dt == data_type::bf16)
        scratchpad.book<float>(key_conv_bias_bf16_convert_wsp, jcp.ngroups);
}

template <cpu_isa_t isa, data_type_t kernel_dt>
void jit_uni_dw_conv_bwd_weights_kernel<isa, kernel_dt>::balance(
        jit_conv_conf_t &jcp, int nthreads) {
    jcp.nthr_oh = jcp.nthr_g = jcp.nthr_mb = 1;
    if (jcp.harness == harness_mb_reduction) {
        /* Basic-Heuristics for parallel strategy:
         * 1) Tries to parallel on the number of Groups (g) where tasks are
         * independent. Otherwise,
         * 2) Tries to split the work across g and MiniBatch (mb).
         * Parallelizing on mb requires computing a reduction for weights.
         *
         * NOTE: because of 'task partitioning' scheme, there will be unbalanced
         * per-thread load when the number of threads is high (e.g. > 16).
         */
        jcp.oh_blk_size = 15;
        jcp.nthr_g = nstl::min(jcp.nb_ch, nthreads);
        jcp.nthr_mb = nstl::min(nstl::max(1, nthreads / jcp.nthr_g), jcp.mb);
        jcp.nthr = jcp.nthr_g * jcp.nthr_mb;
    } else if (jcp.harness == harness_nxc) {
        /* Allocate threads and partition space with regards to 'nb_ch', 'mb'
         * and 'nb_oh' (derived from selecting 'oh_block')
         *
         * note: 'prioritize_threading == true' showed slightly greater
         * performance, but there might be cases where the opposite holds true;
         * code is left for future tuning. */
        partition_nthr_nxc(jcp, nthreads, true);
        jcp.nthr = jcp.nthr_g * jcp.nthr_mb * jcp.nthr_oh;
    }
}

template <cpu_isa_t isa, data_type_t kernel_dt>
void jit_uni_dw_conv_bwd_weights_kernel<isa, kernel_dt>::partition_nthr_nxc(
        jit_conv_conf_t &jcp, int nthreads, bool prioritize_threading) {

    /* Explore thread partitioning space across 'nb_ch', 'mb' and 'nb_oh'
     * (determined by 'oh / oh_block'). Prioritize finding a
     * partition where the most number of threads are used ('thr_eff').
     *
     * Additionally, try to reduce work imbalance across threads
     * (i.e. 'total_imbalance').
     */
    float best_thr_eff = 0.; // maximinze
    float best_imbalance = 1.; // minimize

    // Performance-tuning variables - enable through 'getenv_int()'
    // if necessary
    const int env_max_nthr_g = nthreads; // DNNL_MAX_NTHR_G
    const int env_max_nthr_mb = nthreads; // DNNL_MAX_NTHR_MB
    const int env_max_nthr_oh = nthreads; // DNNL_MAX_NTHR_OH
    const int env_min_oh_block = 1; // DNNL_MIN_OH_BLOCK

    const int ch_outer_blocks = utils::div_up(jcp.nb_ch, jcp.nb_ch_blocking);
    int max_g = nstl::min(env_max_nthr_g, nstl::min(ch_outer_blocks, nthreads));
    for (int g = max_g; g >= 1; --g) {
        int cur_nthr_g = g;
        auto div_nthr_g = nthreads / cur_nthr_g;

        int available_nthr_mb = div_nthr_g;
        int max_mb = nstl::min(
                env_max_nthr_mb, nstl::min(jcp.mb, available_nthr_mb));
        for (int mb = max_mb; mb >= 1; --mb) {
            int cur_nthr_mb = mb;
            auto div_nthr_mb = available_nthr_mb / cur_nthr_mb;

            // used to skip cases where efficiency can only worsen
            bool prev_under_blocked = false;

            int available_nthr_oh = nstl::min(
                    jcp.oh, nstl::min(env_max_nthr_oh, div_nthr_mb));
            int max_oh_block = jcp.oh;
            // Note: maybe it's worth exploring a heuristic to determine
            // optimal_min(oh_block)
            int min_oh_block
                    = nstl::max(1, nstl::min(jcp.oh, env_min_oh_block));
            for (int oh_block = max_oh_block; oh_block >= min_oh_block;
                    --oh_block) {

                // Calculate most efficient approximation for thread use and/or
                // blocking:
                int approx_g_block = utils::div_up(ch_outer_blocks, cur_nthr_g);
                int approx_mb_block = utils::div_up(jcp.mb, cur_nthr_mb);
                int approx_oh_block = utils::div_up(jcp.oh, oh_block);

                int cur_nthr_oh = nstl::min(available_nthr_oh, approx_oh_block);

                // calculate thread use efficiency
                int total_nthr = cur_nthr_g * cur_nthr_mb * cur_nthr_oh;
                float thr_eff = ((float)total_nthr) / nthreads;
                assert(total_nthr <= nthreads);

                // efficiency can only worsen, skip
                if (prev_under_blocked && available_nthr_oh < approx_oh_block) {
                    break;
                }

                // calculate imbalance
                float imbalance_g = ((float)std::abs(approx_g_block * cur_nthr_g
                                            - ch_outer_blocks))
                        / ch_outer_blocks;
                float imbalance_mb
                        = ((float)std::abs(
                                  approx_mb_block * cur_nthr_mb - jcp.mb))
                        / jcp.mb;
                float imbalance_oh
                        = ((float)std::abs(oh_block * cur_nthr_oh - jcp.oh))
                        / jcp.oh;
                float total_imbalance = imbalance_g * (jcp.mb * jcp.oh)
                        + imbalance_mb * (ch_outer_blocks * jcp.oh)
                        + imbalance_oh * (ch_outer_blocks * jcp.mb);

                /* 1) When 'prioritize_threading == true'
                 * First Condition: pick the blocking strategy that uses the
                 * most threads.
                 * Second Condition: if current blocking strategy uses at least
                 * the same amount of threads than the previous best (or more),
                 * chose if work imbalance is less than previous best.
                 *
                 * 2) Otherwise, ('prioritize_threading == false')
                 * First Condition: pick the blocking strategy that has the
                 * lowest thread work imbalance.
                 * Second Condition: if current blocking strategy has at least
                 * the same amount of work imbalance than the previous best(or
                 * lower), chose if it has more number of threads working.
                 * */
                const bool first_condition = prioritize_threading
                        ? best_thr_eff <= thr_eff
                        : best_imbalance >= total_imbalance;
                const bool second_condition = prioritize_threading
                        ? best_thr_eff == thr_eff
                                && best_imbalance <= total_imbalance
                        : best_imbalance == total_imbalance
                                && best_thr_eff >= thr_eff;
                if (first_condition) {
                    if (second_condition) { continue; }
                    jcp.nthr_g = cur_nthr_g;
                    jcp.nthr_mb = cur_nthr_mb;
                    jcp.nthr_oh = cur_nthr_oh;
                    jcp.oh_blk_size = oh_block;
                    best_imbalance = total_imbalance;
                    best_thr_eff = thr_eff;
                }
                prev_under_blocked = oh_block * cur_nthr_oh < jcp.oh;
            }
        }
    }
}

template struct jit_uni_dw_conv_fwd_kernel<avx512_core, bf16>;
template struct jit_uni_dw_conv_fwd_kernel<avx512_common, f32>;
template struct jit_uni_dw_conv_fwd_kernel<avx2, f32>;
template struct jit_uni_dw_conv_fwd_kernel<sse41, f32>;

template struct jit_uni_dw_conv_bwd_data_kernel<avx512_core, bf16>;
template struct jit_uni_dw_conv_bwd_data_kernel<avx512_common, f32>;
template struct jit_uni_dw_conv_bwd_data_kernel<avx2, f32>;
template struct jit_uni_dw_conv_bwd_data_kernel<sse41, f32>;

template struct jit_uni_dw_conv_bwd_weights_kernel<avx512_core, bf16>;
template struct jit_uni_dw_conv_bwd_weights_kernel<avx512_common, f32>;
template struct jit_uni_dw_conv_bwd_weights_kernel<avx2, f32>;
template struct jit_uni_dw_conv_bwd_weights_kernel<sse41, f32>;
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
