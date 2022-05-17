/*******************************************************************************
* Copyright 2018-2021 Intel Corporation
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

#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "cpu/cpu_primitive.hpp"
#include "cpu/zero_point_utils.hpp"

#include "cpu/x64/jit_avx512_core_x8s8s32x_deconvolution.hpp"

#define GET_OFF(field) offsetof(jit_deconv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;
using namespace Xbyak;

using namespace nstl;

#define wht_blk_off(d, g, ...) \
    (pd()->with_groups() ? (d).blk_off((g), __VA_ARGS__) \
                         : (d).blk_off(__VA_ARGS__))

template <typename Vmm>
jit_avx512_core_x8s8s32x_deconv_fwd_kernel<Vmm>::
        jit_avx512_core_x8s8s32x_deconv_fwd_kernel(const jit_conv_conf_t &ajcp,
                const primitive_attr_t &attr, const memory_desc_t &dst_md)
    : jcp(ajcp), attr_(attr), postops_injector_(nullptr) {

    if (jcp.with_eltwise || jcp.with_binary || jcp.with_sum || jcp.with_depthwise || jcp.with_quantization) {
        const std::size_t tail_size = jcp.is_depthwise
                ? jcp.ngroups % jcp.ch_block
                : jcp.oc_without_padding % jcp.oc_block;

        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = true;
        static constexpr bool use_exact_tail_scalar_bcast = false;

        const binary_injector::rhs_arg_static_params_t rhs_sp {
                static_cast<size_t>(Xbyak::Xmm(31).getIdx()), this->rdx,
                this->r14, preserve_gpr, preserve_vmm,
                GET_OFF(post_ops_binary_rhs_arg_vec),
                memory_desc_wrapper(dst_md), tail_size, ktail_mask,
                use_exact_tail_scalar_bcast};
        const binary_injector::static_params_t bsp {this->param1, rhs_sp};

        if (jcp.ver == ver_vnni) {
            vmm_d_weights = Vmm(28);
            vmm_d_bias = Vmm(29);
        } else {
            vmm_d_weights = Vmm(26);
            vmm_d_bias = Vmm(27);
        }

        const quantization_injector::static_params_t qsp {vmm_d_weights.getIdx(), vmm_d_bias.getIdx(), reg_d_weights, reg_d_bias};

        postops_injector_ = utils::make_unique<
                injector::jit_uni_postops_injector_t<avx512_core>>(
                this, jcp.post_ops, bsp, qsp);
    }
}

template <typename Vmm>
jit_avx512_core_x8s8s32x_deconv_fwd_kernel<
        Vmm>::~jit_avx512_core_x8s8s32x_deconv_fwd_kernel()
        = default;

status_t _jit_avx512_core_x8s8s32x_deconv_fwd_kernel::init_conf(
        jit_conv_conf_t &jcp, const deconvolution_desc_t &cd,
        memory_desc_t &src_md, memory_desc_t &weights_md, memory_desc_t &dst_md,
        const bool with_bias, memory_desc_t &bias_md, primitive_attr_t &attr,
        int nthreads) {
    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper bias_d(&bias_md);

    if (!(mayiuse(avx512_core)
                && one_of(src_d.data_type(), data_type::u8, data_type::s8)
                && weights_d.data_type() == data_type::s8
                && one_of(dst_d.data_type(), data_type::f32, data_type::s32,
                        data_type::s8, data_type::u8)))
        return status::unimplemented;

    jcp = zero<decltype(jcp)>();
    jcp.nthr = nthreads;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    jcp.signed_input = src_d.data_type() == data_type::s8;
    const int ndims = jcp.ndims = dst_d.ndims();
    const bool is_1d = ndims == 3;
    const bool is_2d = ndims == 4;
    const bool is_3d = ndims == 5;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.id = is_3d ? src_d.dims()[2] : 1;
    jcp.oc_without_padding = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = src_d.dims()[1] / jcp.ngroups;
    jcp.is_depthwise = true && with_groups
            && utils::everyone_is(
                    1, jcp.ic_without_padding, jcp.oc_without_padding);

    /* TODO: future work, on hold until depthwise specialized kernel is
     * implemented. */
    if (jcp.is_depthwise && (jcp.signed_input || is_3d))
        return status::unimplemented;

    if (!zero_points_valid(&attr)) return status::unimplemented;
    jcp.src_zero_point = !attr.zero_points_.has_default_values(DNNL_ARG_SRC);
    jcp.dst_zero_point = !attr.zero_points_.has_default_values(DNNL_ARG_DST);
    jcp.zp_src_is_common = attr.zero_points_.common(DNNL_ARG_SRC);

    format_tag_t dat_tag = utils::pick(
            ndims - 3, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);

    if (src_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(src_md, dat_tag));
        jcp.src_tag = dat_tag;
    } else {
        jcp.src_tag = src_d.matches_one_of_tag(dat_tag);
    }
    if (jcp.src_tag != dat_tag) return status::unimplemented;

    if (dst_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(dst_md, dat_tag));
        jcp.dst_tag = dat_tag;
    } else {
        jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag);
    }
    if (jcp.dst_tag != dat_tag) return status::unimplemented;

    auto set_or_check_wei_format = [&]() {
        using namespace format_tag;
        format_tag_t wei_tag;
        if (jcp.ic_block == 16 || jcp.ch_block == 16) {
            if (is_3d) {
                wei_tag = with_groups ? gOIdhw4i16o4i : OIdhw4i16o4i;
            } else if (is_1d) {
                wei_tag = with_groups ? jcp.is_depthwise ? Goiw16g : gOIw4i16o4i
                                      : OIw4i16o4i;
            } else {
                assert(is_2d);
                wei_tag = with_groups
                        ? jcp.is_depthwise ? Goihw16g : gOIhw4i16o4i
                        : OIhw4i16o4i;
            }
        } else if (jcp.ic_block == 8) {
            assert(with_groups);
            wei_tag = is_3d ? gOIdhw2i8o4i : is_2d ? gOIhw2i8o4i : gOIw2i8o4i;
        } else {
            assert(with_groups && jcp.ic_block == 4);
            wei_tag = is_3d ? gOIdhw4o4i : is_2d ? gOIhw4o4i : gOIw4o4i;
        }

        memory_desc_t want_wei_md = weights_md;
        memory_desc_init_by_tag(want_wei_md, wei_tag);
        if (jcp.signed_input && !jcp.is_depthwise) {
            want_wei_md.extra.flags = 0
                    | memory_extra_flags::compensation_conv_s8s8;
            want_wei_md.extra.compensation_mask = (1 << 0)
                    + (with_groups && !jcp.is_depthwise ? (1 << 1) : 0);
            want_wei_md.extra.scale_adjust = 1.f;
        }
        if (jcp.src_zero_point) set_zp_src_comp_flags(want_wei_md, with_groups);

        if (weights_md.format_kind == format_kind::any) {
            weights_md = want_wei_md;
            return true;
        }

        return weights_md == want_wei_md;
    };

    jcp.with_bias = with_bias;
    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, format_tag::x));
    }

    jcp.prop_kind = cd.prop_kind;
    jcp.mb = src_d.dims()[0];
    jcp.ih = is_1d ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = is_3d ? dst_d.dims()[2] : 1;
    jcp.oh = is_1d ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];
    jcp.kd = is_3d ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = is_1d ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];
    jcp.f_pad = is_3d ? cd.padding[0][0] : 0;
    jcp.t_pad = is_1d ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.stride_d = is_3d ? cd.strides[0] : 1;
    jcp.stride_h = is_1d ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    if (jcp.is_depthwise) {
        jcp.ch_block = 16;
        jcp.oc_block = 1;
        jcp.ic_block = 1;
    } else {
        jcp.ch_block = 1;
        jcp.oc_block = 16;
        jcp.ic_block = 16;

        if (jcp.ngroups == 1) {
            jcp.oc = utils::rnd_up(jcp.oc_without_padding, jcp.oc_block);
            jcp.ic = utils::rnd_up(jcp.ic_without_padding, jcp.ic_block);
        } else if (jcp.ngroups != 1
                && ((jcp.ic % jcp.ic_block != 0)
                        || (jcp.oc % jcp.oc_block != 0))) {
            /* For grouped deconvolutions, oneDNN doesn't support padding.
                When channels per group is not multiple of 16:
                - Use Ymm when channels per group is multiple of 8,
                - Use Xmm when channels per group is multiple of 4,
                - Otherwise return unimplemented. */
            jcp.ic_block = (jcp.ic % 8 == 0) && (jcp.oc % 8 == 0) ? 8 : 4;
            jcp.oc_block = jcp.ic_block;
        }
        if (jcp.ic % jcp.ic_block != 0 || jcp.oc % jcp.oc_block != 0)
            return status::unimplemented;
    }

    if (!set_or_check_wei_format()) return status::unimplemented;

    jcp.dilate_d = is_3d ? cd.dilates[0] : 0;
    jcp.dilate_h = is_1d ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    if (!IMPLICATION(jcp.dilate_d, jcp.stride_d == 1)
            || !IMPLICATION(jcp.dilate_h, jcp.stride_h == 1)
            || !IMPLICATION(jcp.dilate_w, jcp.stride_w == 1))
        return status::unimplemented;

    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.iw, jcp.ow, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.ih, jcp.oh, jcp.stride_h, ext_kh);
    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.id, jcp.od, jcp.stride_d, ext_kd);
    bool kernel_outside_src = false || ext_kw <= jcp.l_pad
            || ext_kw <= jcp.r_pad || ext_kh <= jcp.t_pad || ext_kh <= jcp.b_pad
            || ext_kd <= jcp.f_pad || ext_kd <= jcp.back_pad;
    if (kernel_outside_src) return status::unimplemented;

    CHECK(attr.set_default_formats(&dst_md));
    if (!post_ops_ok(jcp, attr, dst_d)) return status::unimplemented;

    const auto &p = attr.post_ops_;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) jcp.eltwise = p.entry_[eltwise_ind].eltwise;
    const int binary_ind = p.find(primitive_kind::binary);
    jcp.with_binary = binary_ind != -1;
    const int depthwise_ind = p.find(primitive_kind::depthwise);
    jcp.with_depthwise = depthwise_ind != -1;
    const int quantization_ind = p.find(primitive_kind::quantization);
    jcp.with_quantization = quantization_ind != -1;

    const int sum_ind = p.find(primitive_kind::sum);
    jcp.with_sum = sum_ind != -1;

    //save post_ops desc for further usage
    jcp.post_ops = p;

    jcp.ver = ver_avx512_core;
    if (mayiuse(avx512_core_vnni)) jcp.ver = ver_vnni;
    int max_regs = jcp.ver == ver_vnni ? 30 : 28;

    if (jcp.with_depthwise || jcp.with_quantization) {
        max_regs -= 2;
    }

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    // only common and per-oc-channel scales are supported
    const bool oscales_ok = one_of(oscales.mask_, 0, 1 << 1);
    if (!oscales_ok) return status::unimplemented;

    jcp.dst_dt = dst_d.data_type();
    jcp.bia_dt = jcp.with_bias ? bias_d.data_type() : data_type::undef;
    jcp.typesize_bia
            = jcp.with_bias ? types::data_type_size(bias_d.data_type()) : 0;
    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());

    jcp.nb_ch = div_up(jcp.ngroups, jcp.ch_block);
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    if (jcp.nb_oc == 0) return status::unimplemented;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    /* kernel blocking params */
    const int regs = max_regs;
    jcp.nb_ch_blocking = 1;
    jcp.nb_oc_blocking = nstl::min(4, jcp.nb_oc);
    for (; jcp.nb_oc_blocking > 1; jcp.nb_oc_blocking--)
        if (jcp.nb_oc % jcp.nb_oc_blocking == 0
                && jcp.l_pad <= regs / (jcp.nb_oc_blocking + 1))
            break;

    jcp.ur_w = regs / (jcp.nb_oc_blocking + 1);
    int l_overflow = max(
            0, ((jcp.kw - 1) * (jcp.dilate_w + 1) - jcp.l_pad) / jcp.stride_w);

    if (jcp.ow < jcp.ur_w) {
        jcp.ur_w = jcp.ow;
        jcp.ur_w_tail = 0;
    } else {
        for (; jcp.ur_w >= 1; jcp.ur_w--) {
            /* ur_w should be multiple of stride_w in order
               to simplify logic for get_ow_start and get_ow_end */
            bool is_multiple_of_stride = jcp.ur_w % jcp.stride_w == 0;

            /* boundary conditions:
               These conditions ensure all elements close to boundary
               are computed in a single call of compute loop */
            bool left_boundary_covered = jcp.ur_w >= l_overflow * jcp.stride_w;
            jcp.ur_w_tail = jcp.ow % jcp.ur_w;
            int r_overflow_no_tail = max(0,
                    ((jcp.kw - 1) * (jcp.dilate_w + 1) - max(0, jcp.r_pad)
                            - jcp.ur_w_tail)
                            / jcp.stride_w);
            bool right_boundary_covered
                    = jcp.ur_w >= r_overflow_no_tail * jcp.stride_w;

            if (is_multiple_of_stride && left_boundary_covered
                    && right_boundary_covered)
                break;
            else if (jcp.ur_w == 1)
                /* The boundary conditions above are also important
                   to maintain simplicity of calls to icb_loop,
                   if those conditions are not satisfied,
                   then special cases will need to be added
                   to use correct l_overflow/r_overflow values
                   when different iterations of compute loop
                   work on the locations close to boundary.
                   So to keep code simple, return unimplemented
                   for extreme case when a good ur_w cannot be found.
                 */
                return status::unimplemented;
        }
    }

    jcp.wei_adj_scale
            = (weights_d.extra().flags & memory_extra_flags::scale_adjust)
            ? weights_d.extra().scale_adjust
            : 1.f;

    jcp.loop_order = jcp.ngroups > 1 ? loop_ngc : loop_cgn;
    return status::success;
}

bool _jit_avx512_core_x8s8s32x_deconv_fwd_kernel::post_ops_ok(
        jit_conv_conf_t &jcp, primitive_attr_t &attr,
        const memory_desc_wrapper &dst_d) {

    using namespace injector;
    const auto &post_ops = attr.post_ops_;
    static constexpr bool sum_at_pos_0_only = true;
    static constexpr bool sum_requires_scale_one = true;

    return injector::post_ops_ok({avx512_core, {eltwise, binary, sum, depthwise, quantization}, post_ops,
            &dst_d, sum_at_pos_0_only, sum_requires_scale_one});
}

void _jit_avx512_core_x8s8s32x_deconv_fwd_kernel::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp,
        const primitive_attr_t &attr) {
    if (jcp.signed_input && jcp.ver != ver_vnni) {
        dim_t count = nstl::max<dim_t>(attr.output_scales_.count_, 16);
        scratchpad.book<float>(key_conv_adjusted_scales, count);
    }

    if (zp::should_calculate_deconv_zp_src_pad_str_comp(jcp)) {
        const dim_t zp_pad_comp_size = jcp.oc_without_padding * jcp.ngroups
                * jcp.kd * jcp.kh * jcp.kw;
        scratchpad.book<int32_t>(key_deconv_zp, zp_pad_comp_size);
    }
}

template <typename Vmm>
void jit_avx512_core_x8s8s32x_deconv_fwd_kernel<Vmm>::compute(
        const Vmm &vreg_acc, const Vmm &vreg_wei, const Vmm &vreg_src) {

    if (jcp.ver == ver_vnni) {
        vpdpbusd(vreg_acc, vreg_src, vreg_wei);
    } else if (jcp.is_depthwise) {
        uni_vmovups(vmm_tmp, vreg_src);
        uni_vpmulld(vmm_tmp, vmm_tmp, vreg_wei);
        uni_vpaddd(vreg_acc, vreg_acc, vmm_tmp);
    } else {
        uni_vpmaddubsw(vmm_tmp, vreg_src, vreg_wei);
        uni_vpmaddwd(vmm_tmp, vmm_tmp, vmm_one);
        uni_vpaddd(vreg_acc, vreg_acc, vmm_tmp);
    }
}

template <typename Vmm>
std::function<Vmm()> jit_avx512_core_x8s8s32x_deconv_fwd_kernel<
        Vmm>::prepare_round_robin_vmm_inp_generator(int ur_w) const noexcept {
    const int start_vmm_idx = vmm_inp(0, jcp.nb_oc_blocking).getIdx();
    const int end_vmm_idx = vmm_inp(ur_w - 1, jcp.nb_oc_blocking).getIdx() + 1;
    int current_vmm_idx = start_vmm_idx;

    return [=]() mutable {
        const Vmm vmm {static_cast<int>(current_vmm_idx++)};

        if (current_vmm_idx == end_vmm_idx) current_vmm_idx = start_vmm_idx;

        return vmm;
    };
}

template <typename Vmm>
void jit_avx512_core_x8s8s32x_deconv_fwd_kernel<Vmm>::apply_zp_src_pad_str_comp(
        int ur_w, int l_overflow, int r_overflow, bool h_padded) {
    Xbyak::Label end_zp_pad, no_tail;

    // apply once per icb loop, zp src stride padding compensation calculated as
    // zp_pad_str_compensation = conv(1, weights_s8) * zero_point_source
    cmp(reg_icb, jcp.nb_ic);
    jne(end_zp_pad, T_NEAR);

    if (jcp.ngroups % jcp.ch_block || jcp.oc_without_padding % jcp.oc_block) {
        if (jcp.is_depthwise)
            cmp(reg_oc_blocks, jcp.nb_ch - 1);
        else
            cmp(reg_oc_blocks, jcp.nb_oc - jcp.nb_oc_blocking);
        jne(no_tail, T_NEAR);

        static constexpr bool last_ocb = true;
        append_zp_src_pad_str_comp(
                ur_w, l_overflow, r_overflow, h_padded, last_ocb);
        jmp(end_zp_pad, T_NEAR);
    }

    L(no_tail);
    static constexpr bool last_ocb = false;

    append_zp_src_pad_str_comp(
            ur_w, l_overflow, r_overflow, h_padded, last_ocb);

    L(end_zp_pad);
}

template <typename Vmm>
void jit_avx512_core_x8s8s32x_deconv_fwd_kernel<
        Vmm>::append_zp_src_pad_str_comp(int ur_w, int l_overflow,
        int r_overflow, bool h_padded, bool last_oc_block) {

    const auto &reg_zp_src_pad_comp = reg_scratch;
    const auto get_next_comp_vmm = prepare_round_robin_vmm_inp_generator(ur_w);
    bool base_comp_addr_loaded = false;

    const auto load_base_zp_src_pad_comp_addr = [&]() {
        if (!base_comp_addr_loaded) {
            if (jcp.ndims == 5) mov(reg_scratch_preserved, reg_scratch);

            if (jcp.ndims > 3)
                mov(reg_zp_src_pad_comp, zp_src_pad_comp_addr);
            else
                mov(reg_zp_src_pad_comp,
                        qword[param1 + GET_OFF(zp_src_pad_str_compensation)]);

            base_comp_addr_loaded = true;
        }
    };

    const auto load_zp_src_pad_comp = [&](const Vmm &zp_pad_comp_vmm,
                                              const Xbyak::Address &comp_addr,
                                              const int ocb) {
        const bool is_last_ocb = last_oc_block && ocb == jcp.nb_oc_blocking - 1;
        const bool is_tail = is_last_ocb && get_tail_size() > 0;
        if (is_tail)
            vmovups(zp_pad_comp_vmm | ktail_mask | T_z, comp_addr);
        else
            vmovups(zp_pad_comp_vmm, comp_addr);
    };

    const auto get_zp_src_comp_pad_off = [&](int it_kw, int ocb) {
        const auto kw_offset = it_kw * jcp.oc_without_padding * jcp.ngroups;
        const auto oc_offset = ocb * jcp.oc_block;

        return (kw_offset + oc_offset) * sizeof(int32_t);
    };

    for (int it_kw = 0; it_kw < jcp.kw; ++it_kw) {
        const int ow_start = get_ow_start(it_kw, l_overflow);
        const int ow_end = get_ow_end(ur_w, it_kw, r_overflow);

        for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
            Vmm zp_src_comp_pad_vmm; // will be assigned later
            bool ocb_zp_loaded = false;

            const auto zp_src_comp_pad_off
                    = get_zp_src_comp_pad_off(it_kw, ocb);

            for (int it_ow = 0; it_ow < ur_w; ++it_ow) {

                const bool inside_padded_area = h_padded
                        || !(it_ow >= ow_start && it_ow < ow_end
                                && ((it_ow + jcp.l_pad - it_kw) % jcp.stride_w
                                        == 0));

                if (inside_padded_area) {
                    load_base_zp_src_pad_comp_addr();

                    if (!ocb_zp_loaded) {
                        zp_src_comp_pad_vmm = get_next_comp_vmm();
                        const auto comp_addr = ptr[reg_zp_src_pad_comp
                                + zp_src_comp_pad_off];
                        load_zp_src_pad_comp(
                                zp_src_comp_pad_vmm, comp_addr, ocb);
                        ocb_zp_loaded = true;
                    }

                    const auto vmm_dst = vmm_out(it_ow, ocb);
                    uni_vpaddd(vmm_dst, vmm_dst, zp_src_comp_pad_vmm);
                }
            }
        }
    }

    if (jcp.ndims > 3) {
        if (!base_comp_addr_loaded) load_base_zp_src_pad_comp_addr();

        const auto kh_offset = jcp.kw * jcp.oc_without_padding * jcp.ngroups
                * sizeof(int32_t);

        add(reg_zp_src_pad_comp, kh_offset);
        mov(zp_src_pad_comp_addr, reg_zp_src_pad_comp);
    }

    if (jcp.ndims == 5 && base_comp_addr_loaded)
        mov(reg_scratch, reg_scratch_preserved);
}

template <typename Vmm>
void jit_avx512_core_x8s8s32x_deconv_fwd_kernel<Vmm>::compute_ker(int ur_w,
        int l_overflow, int r_overflow, ker_block_t last_ic_block_flag,
        bool h_padded) {

    const bool signed_input_or_src_zp
            = (jcp.signed_input || jcp.src_zero_point);

    const int ch_block_all = jcp.ch_block * jcp.ic_block * jcp.oc_block;
    const int ur_w_stride = signed_input_or_src_zp ? 1 : jcp.stride_w;

    auto src_offset = [=](int oj, int icb, int ki) {
        return jcp.typesize_in
                * (((oj + jcp.l_pad - ki * (jcp.dilate_w + 1)) / jcp.stride_w)
                                * jcp.ngroups * jcp.ic_without_padding
                        + icb * 4);
    };

    auto kernel_offset = [=](int ocb, int icb, int ki) {
        return jcp.typesize_in
                * ((ocb * jcp.nb_ic * jcp.kd * jcp.kh * jcp.kw + ki)
                                * ch_block_all
                        + icb * jcp.oc_block * ic_sub_step);
    };

    for (int ki = 0; ki < jcp.kw; ki++) {
        int jj_start = get_ow_start(ki, l_overflow);
        int jj_end = get_ow_end(ur_w, ki, r_overflow);

        int _start = (signed_input_or_src_zp) ? 0 : jj_start;
        int _end = (signed_input_or_src_zp) ? ur_w : jj_end;

        int tail_size = jcp.is_depthwise ? jcp.ngroups % jcp.ch_block
                                         : jcp.ic_without_padding % 4;
        int n_ic_blocks = jcp.is_depthwise
                ? 1
                : (last_ic_block_flag & ~no_last_block ? div_up(
                           jcp.ic_without_padding % jcp.ic_block, 4)
                                                       : jcp.ic_block / 4);

        for (int icb1 = 0; icb1 < n_ic_blocks; icb1++) {
            if (h_padded == true) {
                if (jcp.signed_input) {
                    /* fill padded area with shifted values */
                    const Vmm inp = vmm_inp(0, jcp.nb_oc_blocking);
                    vpxord(inp, inp, inp);
                    vpsubb(inp, inp, vmm_shift);
                }
            } else {

                for (int jj = _start; jj < _end; jj += ur_w_stride) {

                    int aux_src_off = src_offset(jj, icb1, ki);

                    if (jj >= jj_start && jj < jj_end
                            && ((jj + jcp.l_pad - ki) % jcp.stride_w == 0)) {
                        if (jcp.is_depthwise) {
                            Vmm vmm_src = vmm_inp(jj, jcp.nb_oc_blocking);
                            if (tail_size != 0) {
                                assert(jcp.nb_oc_blocking == 1);
                                vmm_src = vmm_src | ktail_mask | T_z;
                            }
                            vpmovzxbd(vmm_src,
                                    EVEX_compress_addr(
                                            aux_reg_src, aux_src_off));
                        } else if ((last_ic_block_flag & last_sp_block)
                                && tail_size != 0 && icb1 == n_ic_blocks - 1) {
                            const Xmm xmm_tmp = Xmm(
                                    vmm_inp(jj, jcp.nb_oc_blocking).getIdx());
                            for (int r = 0; r < tail_size; ++r)
                                vpinsrb(xmm_tmp, xmm_tmp,
                                        ptr[aux_reg_src + aux_src_off + r], r);
                            vpbroadcastd(
                                    vmm_inp(jj, jcp.nb_oc_blocking), xmm_tmp);
                        } else {
                            vpbroadcastd(vmm_inp(jj, jcp.nb_oc_blocking),
                                    EVEX_compress_addr(
                                            aux_reg_src, aux_src_off));
                        }
                        if (jcp.signed_input)
                            vpsubb(vmm_inp(jj, jcp.nb_oc_blocking),
                                    vmm_inp(jj, jcp.nb_oc_blocking), vmm_shift);
                    } else {
                        /* fill padded area with shifted values */
                        if (jcp.signed_input) {
                            const Vmm inp = vmm_inp(jj, jcp.nb_oc_blocking);
                            vpxord(inp, inp, inp);
                            vpsubb(inp, inp, vmm_shift);
                        }
                    }
                }
            }
            for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
                int aux_filt_off = kernel_offset(ocb, icb1, ki);

                if (_end - _start > 0) {
                    if (jcp.is_depthwise)
                        vpmovsxbd(vmm_wei,
                                EVEX_compress_addr(aux_reg_filt, aux_filt_off));
                    else
                        vmovups(vmm_wei,
                                EVEX_compress_addr(aux_reg_filt, aux_filt_off));
                }
                for (int jj = _start; jj < _end; jj += ur_w_stride) {
                    const bool jj_between_start_end
                            = jj >= jj_start && jj < jj_end;
                    const bool ki_applies_to_stride
                            = (jj + jcp.l_pad - ki) % jcp.stride_w == 0;
                    const bool inside_padded_area = h_padded
                            || !(jj_between_start_end && ki_applies_to_stride);
                    const auto vmm_dst = vmm_out(jj, ocb);
                    if (jcp.signed_input || !inside_padded_area) {
                        const Vmm inp = vmm_inp(
                                h_padded ? 0 : jj, jcp.nb_oc_blocking);
                        compute(vmm_dst, vmm_wei, inp);
                    }
                }
            }
        }
    }

    if (zp::should_calculate_deconv_zp_src_pad_str_comp(jcp))
        apply_zp_src_pad_str_comp(ur_w, l_overflow, r_overflow, h_padded);
}

template <typename Vmm>
void jit_avx512_core_x8s8s32x_deconv_fwd_kernel<Vmm>::kh_loop(int ur_w,
        int l_overflow, int r_overflow, ker_block_t last_ic_block_flag) {

    const bool signed_input_or_src_zp
            = (jcp.signed_input || jcp.src_zero_point);

    int ch_block_all = jcp.ch_block * jcp.ic_block * jcp.oc_block;
    int shift_src_ih = jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw
            * jcp.ngroups * jcp.ic_without_padding;
    int shift_src_id = jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * jcp.iw
            * jcp.ngroups * jcp.ic_without_padding;
    const int stride_h = signed_input_or_src_zp ? 1 : jcp.stride_h;
    int shift_filt_kh = jcp.typesize_in * jcp.kw * ch_block_all * stride_h;
    const int stride_d = signed_input_or_src_zp ? 1 : jcp.stride_d;
    int shift_filt_kd
            = jcp.typesize_in * jcp.kw * ch_block_all * jcp.kh * stride_d;

    Label kd_loop_label, kh_loop_label, skip_kh_loop, skip_kd_loop;
    Label t_overflow_label, no_t_overflow_label, b_overflow_label,
            no_b_overflow_label;
    Label back_overflow_label, no_back_overflow_label, d_h_overflow_label,
            front_overflow_label, no_front_overflow_label, d_h_overflow_label2;
    if (jcp.ndims == 5) {
        mov(aux_reg_filt_d, reg_filt);
        mov(aux_reg_src_d, reg_src);

        if (signed_input_or_src_zp) {
            mov(reg_ki, ptr[param1 + GET_OFF(back_overflow)]);
            cmp(reg_ki, 0);
            je(no_back_overflow_label, T_NEAR);
            L(back_overflow_label);
            {
                mov(aux_reg_filt, aux_reg_filt_d);
                mov(reg_kh, jcp.kh);
                L(d_h_overflow_label);
                {
                    compute_ker(ur_w, 0, 0, last_ic_block_flag, true);
                    add(aux_reg_filt, shift_filt_kh);
                    dec(reg_kh);
                    jnz(d_h_overflow_label);
                }

                add(aux_reg_filt_d, shift_filt_kd);
                dec(reg_ki);
                jnz(back_overflow_label);
            }
            L(no_back_overflow_label);
        }

        mov(reg_ki, ptr[param1 + GET_OFF(kd_padding)]);

        if ((signed_input_or_src_zp) || (jcp.dilate_d >= jcp.id)
                || ((!signed_input_or_src_zp)
                        && ((min(jcp.f_pad, jcp.back_pad) < 0)
                                || ((jcp.kd - 1) * (jcp.dilate_d + 1)
                                        < nstl::max(
                                                jcp.f_pad, jcp.back_pad))))) {
            cmp(reg_ki, 0);
            je(skip_kd_loop, T_NEAR);
        }

        L(kd_loop_label);
        mov(aux_reg_src, aux_reg_src_d);
        mov(aux_reg_filt, aux_reg_filt_d);
    } else {
        mov(aux_reg_src, reg_src);
        mov(aux_reg_filt, reg_filt);
    }

    if (signed_input_or_src_zp && jcp.ndims > 3) {
        /* Weights are transposed, so first compute 'bottom' padding. */
        mov(reg_overflow, ptr[param1 + GET_OFF(b_overflow)]);
        cmp(reg_overflow, 0);
        je(no_b_overflow_label, T_NEAR);
        L(b_overflow_label);
        {
            compute_ker(ur_w, 0, 0, last_ic_block_flag, true);

            add(aux_reg_filt, shift_filt_kh);
            dec(reg_overflow);
            cmp(reg_overflow, 0);
            jg(b_overflow_label, T_NEAR);
        }
        L(no_b_overflow_label);
    }

    mov(reg_kh, ptr[param1 + GET_OFF(kh_padding)]);

    if ((signed_input_or_src_zp) || (jcp.dilate_h >= jcp.ih)
            || ((!signed_input_or_src_zp)
                    && ((min(jcp.t_pad, jcp.b_pad) < 0)
                            || ((jcp.kh - 1) * (jcp.dilate_h + 1)
                                    < nstl::max(jcp.t_pad, jcp.b_pad))))) {
        cmp(reg_kh, 0);
        je(skip_kh_loop, T_NEAR);
    }

    L(kh_loop_label);
    {
        compute_ker(ur_w, l_overflow, r_overflow, last_ic_block_flag, false);
        sub(aux_reg_src, shift_src_ih);
        add(aux_reg_filt, shift_filt_kh);
        dec(reg_kh);

        /* Insert weight compensation in stride 'holes' */
        if (signed_input_or_src_zp && jcp.stride_h > 1) {
            Label kh_comp_loop;

            cmp(reg_kh, 0);
            je(skip_kh_loop, T_NEAR);
            mov(reg_comp_strides, jcp.stride_h - 1);
            L(kh_comp_loop);
            {
                compute_ker(ur_w, 0, 0, last_ic_block_flag, true);
                add(aux_reg_filt, shift_filt_kh);
                dec(reg_comp_strides);
                cmp(reg_comp_strides, 0);
                jg(kh_comp_loop, T_NEAR);
            }
        }
        cmp(reg_kh, 0);
        jg(kh_loop_label, T_NEAR);
    }
    L(skip_kh_loop);
    if (signed_input_or_src_zp && jcp.ndims > 3) {
        mov(reg_overflow, ptr[param1 + GET_OFF(t_overflow)]);
        cmp(reg_overflow, 0);
        je(no_t_overflow_label, T_NEAR);
        L(t_overflow_label);
        {
            compute_ker(ur_w, 0, 0, last_ic_block_flag, true);

            add(aux_reg_filt, shift_filt_kh);
            dec(reg_overflow);
            cmp(reg_overflow, 0);
            jg(t_overflow_label, T_NEAR);
        }
        L(no_t_overflow_label);
    }

    if (jcp.ndims == 5) {
        sub(aux_reg_src_d, shift_src_id);
        add(aux_reg_filt_d, shift_filt_kd);
        dec(reg_ki);

        /* Insert weight compensation in stride 'holes' */
        if (signed_input_or_src_zp && jcp.stride_d > 1) {
            Label kd_comp_loop, kd_kh_comp_loop;
            cmp(reg_ki, 0);
            jz(skip_kd_loop, T_NEAR);
            mov(reg_comp_strides, jcp.stride_d - 1);
            L(kd_comp_loop);
            mov(aux_reg_filt, aux_reg_filt_d);
            mov(reg_kh, jcp.kh);
            L(kd_kh_comp_loop);
            {
                compute_ker(ur_w, 0, 0, last_ic_block_flag, true);
                add(aux_reg_filt, shift_filt_kh);
                dec(reg_kh);
                jnz(kd_kh_comp_loop, T_NEAR);
            }
            add(aux_reg_filt_d, shift_filt_kd);
            dec(reg_comp_strides);
            jnz(kd_comp_loop);
        }

        cmp(reg_ki, 0);
        jg(kd_loop_label, T_NEAR);
        L(skip_kd_loop);
        if (signed_input_or_src_zp) {
            mov(reg_ki, ptr[param1 + GET_OFF(f_overflow)]);
            cmp(reg_ki, 0);
            jz(no_front_overflow_label, T_NEAR);
            L(front_overflow_label);
            {
                mov(aux_reg_filt, aux_reg_filt_d);
                mov(reg_kh, jcp.kh);
                L(d_h_overflow_label2);
                {
                    compute_ker(ur_w, 0, 0, last_ic_block_flag, true);
                    add(aux_reg_filt, shift_filt_kh);
                    dec(reg_kh);
                    jnz(d_h_overflow_label2);
                }
                add(aux_reg_filt_d, shift_filt_kd);
                dec(reg_ki);
                jnz(front_overflow_label);
            }
            L(no_front_overflow_label);
        }
    }
}
template <typename Vmm>
int jit_avx512_core_x8s8s32x_deconv_fwd_kernel<Vmm>::get_tail_size() const
        noexcept {
    return jcp.is_depthwise ? jcp.ngroups % jcp.ch_block
                            : jcp.oc_without_padding % jcp.oc_block;
}

template <typename Vmm>
int jit_avx512_core_x8s8s32x_deconv_fwd_kernel<Vmm>::get_blocking_size() const
        noexcept {
    return jcp.is_depthwise ? jcp.ch_block : jcp.oc_block;
}

template <typename Vmm>
void jit_avx512_core_x8s8s32x_deconv_fwd_kernel<Vmm>::prepare_output(int ur_w) {
    for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
        for (int ur = 0; ur < ur_w; ur++) {
            const Vmm vmm = vmm_out(ur, ocb);
            vpxord(vmm, vmm, vmm);
        }
    }
    if (jcp.signed_input) {
        xor_(reg_scratch, reg_scratch);
        Reg8 _t8 = reg_scratch.cvt8();
        mov(_t8, (int8_t)-128);
        vpbroadcastb(vmm_shift, _t8);
    }
}

template <typename Vmm>
void jit_avx512_core_x8s8s32x_deconv_fwd_kernel<Vmm>::cvt2ps(
        data_type_t type_in, Vmm vmm_in, const Operand &op, bool mask_flag) {
    const Vmm vmm = mask_flag ? vmm_in | ktail_mask | T_z : vmm_in;
    switch (type_in) {
        case data_type::f32:
        case data_type::s32: vmovups(vmm, op); break;
        case data_type::s8: vpmovsxbd(vmm, op); break;
        case data_type::u8: vpmovzxbd(vmm, op); break;
        default: assert(!"unsupported data type");
    }
    if (type_in != data_type::f32) vcvtdq2ps(vmm_in, vmm_in);
}

template <typename Vmm>
void jit_avx512_core_x8s8s32x_deconv_fwd_kernel<Vmm>::store_output(
        int ur_w, bool last_oc_block) {
    mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
    mov(reg_ptr_scales, ptr[param1 + GET_OFF(scales)]);

    if (jcp.signed_input)
        mov(reg_compensation, ptr[param1 + GET_OFF(compensation)]);

    if (jcp.src_zero_point) {
        mov(reg_zp_src_, ptr[param1 + GET_OFF(src_zero_point)]);
        mov(reg_zp_compensation, ptr[param1 + GET_OFF(zp_compensation)]);
    }

    const auto &p = attr_.post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    const float *p_sum_scale
            = (sum_idx != -1) ? &p.entry_[sum_idx].sum.scale : nullptr;
    if (p_sum_scale && *p_sum_scale != 1.f)
        mov(reg_ptr_sum_scale, (size_t)p_sum_scale);

    if (jcp.with_bias && jcp.signed_input && jcp.ver != ver_vnni) {
        mov(reg_bias_alpha, float2int(jcp.wei_adj_scale));
        vmovq(xmm_bias_alpha(), reg_bias_alpha);
        vbroadcastss(vmm_bias_alpha(), xmm_bias_alpha());
    }

    if (jcp.src_zero_point) {
        const auto &vmm_src_zp = vmm_tmp;
        const auto &vmm_zp_comp = vmm_wei;
        uni_vbroadcastss(vmm_src_zp, ptr[reg_zp_src_]);

        const bool is_tail = get_tail_size() > 0;
        for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
            const int zp_offset = sizeof(int32_t) * ocb * jcp.oc_block;
            const bool is_last_ocb
                    = last_oc_block && ocb == jcp.nb_oc_blocking - 1;
            const auto vmm = is_last_ocb && is_tail
                    ? vmm_zp_comp | ktail_mask | T_z
                    : vmm_zp_comp;
            vmovups(vmm, ptr[reg_zp_compensation + zp_offset]);

            uni_vpmulld(vmm_zp_comp, vmm, vmm_src_zp);

            for (int ur = 0; ur < ur_w; ur++) {
                const auto vmm_dst = vmm_out(ur, ocb);
                uni_vpaddd(vmm_dst, vmm_dst, vmm_zp_comp);
            }
        }
    }

    for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
        const bool mask_flag = last_oc_block && ocb == jcp.nb_oc_blocking - 1;
        int scale_offset
                = jcp.is_oc_scale * (sizeof(float) * ocb * jcp.oc_block);

        const Vmm vmm_bias = vmm_tmp;
        if (jcp.with_bias) {
            int bias_offset = jcp.typesize_bia * ocb * jcp.oc_block;
            auto bias_addr = EVEX_compress_addr(reg_bias, bias_offset);
            cvt2ps(jcp.bia_dt, vmm_bias, bias_addr, mask_flag);
            if (jcp.signed_input && jcp.ver != ver_vnni)
                vmulps(vmm_bias, vmm_bias, vmm_bias_alpha());
        }
        if (jcp.signed_input) {
            int comp_offset = sizeof(int32_t) * ocb * jcp.oc_block;
            auto comp_addr = EVEX_compress_addr(reg_compensation, comp_offset);
            cvt2ps(data_type::s32, vmm_comp, comp_addr, mask_flag);
        }

        for (int ur = 0; ur < ur_w; ur++) {
            const Vmm vmm = vmm_out(ur, ocb);
            vcvtdq2ps(vmm, vmm);
            if (jcp.signed_input) vaddps(vmm, vmm, vmm_comp);
            if (jcp.with_bias) vaddps(vmm, vmm, vmm_bias);
            const Vmm mask_vmm = mask_flag ? vmm | ktail_mask | T_z : vmm;
            vmulps(mask_vmm, vmm,
                    EVEX_compress_addr(reg_ptr_scales, scale_offset));
        }
    }
    /* Do post-ops */
    if (jcp.with_eltwise || jcp.with_binary || jcp.with_sum || jcp.with_depthwise || jcp.with_quantization) {
        const auto sum_injector = [&]() {
            if (p_sum_scale) { // post_op: sum
                for (int k = 0; k < jcp.nb_oc_blocking; k++) {
                    const bool mask_flag
                            = last_oc_block == 1 && k == jcp.nb_oc_blocking - 1;
                    for (int j = 0; j < ur_w; j++) {
                        int aux_output_offset = jcp.typesize_out
                                * (k * jcp.oc_block
                                        + j * jcp.oc_without_padding
                                                * jcp.ngroups);
                        auto addr = EVEX_compress_addr(
                                reg_dst, aux_output_offset);
                        const Vmm vmm = vmm_out(j, k);
                        cvt2ps(jcp.dst_dt, vmm_prev_dst, addr, mask_flag);
                        if (*p_sum_scale == 1.f)
                            vaddps(vmm, vmm_prev_dst);
                        else
                            vfmadd231ps(vmm, vmm_prev_dst,
                                    zword_b[reg_ptr_sum_scale]);
                    }
                }
            }
        };

        if (p_sum_scale)
            postops_injector_->set_lambda_injector(
                    primitive_kind::sum, sum_injector);

        binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
        if (jcp.with_binary) {
            for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
                const bool mask_flag
                        = last_oc_block && ocb == jcp.nb_oc_blocking - 1;
                for (int ur = 0; ur < ur_w; ur++) {
                    const int vmm_idx = vmm_out(ur, ocb).getIdx();
                    rhs_arg_params.vmm_idx_to_oc_elem_off_addr.emplace(
                            vmm_idx, ptr[param1 + GET_OFF(oc_l_off)]);
                    rhs_arg_params.vmm_idx_to_oc_elem_off_val.emplace(
                            vmm_idx, ocb * jcp.oc_block);
                    if (mask_flag)
                        rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
                }
            }
        }

        std::map<size_t, int> vmm_idx_off;
        for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
            for (int ur = 0; ur < ur_w; ur++) {
                vmm_idx_off.insert({vmm_out(ur, ocb).getIdx(), ocb * jcp.oc_block * sizeof(float)});
            }
        }
        depthwise_injector::dynamic_params_t ddp {vmm_d_weights.getIdx(), vmm_d_bias.getIdx(), reg_d_weights, reg_d_bias,
                                                  ptr[this->param1 + GET_OFF(oc_off)], vmm_idx_off,
                                                  this->rsp, base_post_ops_data_offset};
        quantization_injector::dynamic_params_t qdp {ptr[this->param1 + GET_OFF(oc_off)], vmm_idx_off, jcp.dst_dt,
                                                     this->rsp, base_post_ops_data_offset};

        const int nb_oc_block
                = jcp.is_depthwise ? jcp.nb_ch_blocking : jcp.nb_oc_blocking;
        postops_injector_->compute_vector_range(
                0, nb_oc_block * ur_w, rhs_arg_params, ddp, qdp);
    }

    if (jcp.dst_zero_point) {
        mov(reg_zp_dst_, ptr[param1 + GET_OFF(dst_zero_point)]);
        const auto &vmm_zp_dst = vmm_tmp;
        uni_vbroadcastss(vmm_zp_dst, ptr[reg_zp_dst_]);
        vcvtdq2ps(vmm_zp_dst, vmm_zp_dst);

        for_(int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++)
        for (int ur = 0; ur < ur_w; ur++) {
            const auto vmm_dst = vmm_out(ur, ocb);
            uni_vaddps(vmm_dst, vmm_dst, vmm_zp_dst);
        }
    }

    // Properly saturate the accumulators for integer datatypes

    // No need to saturate on lower bound for signed integer types, as
    // the conversion to int would return INT_MIN, and then proper
    // saturation will happen when storing data
    if (jcp.dst_dt == data_type::u8) {
        vpxord(vmm_zero, vmm_zero, vmm_zero);
        for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
            for (int ur = 0; ur < ur_w; ur++) {
                const Vmm vmm = vmm_out(ur, ocb);
                vmaxps(vmm, vmm_zero, vmm);
            }
        }
    }

    if (one_of(jcp.dst_dt, data_type::u8, data_type::s8, data_type::s32)) {
        float saturation_ubound = types::max_value<float>(jcp.dst_dt);
        Xmm xmm_saturation(vmm_saturation.getIdx());
        mov(reg_ptr_saturation_ubound, float2int(saturation_ubound));
        vmovq(xmm_saturation, reg_ptr_saturation_ubound);
        vbroadcastss(vmm_saturation, xmm_saturation);

        for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
            for (int ur = 0; ur < ur_w; ur++) {
                const Vmm vmm = vmm_out(ur, ocb);
                vminps(vmm, vmm, vmm_saturation);
            }
        }
    }

    if (one_of(jcp.dst_dt, data_type::u8, data_type::s8, data_type::s32)) {
        for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
            for (int ur = 0; ur < ur_w; ur++) {
                const Vmm vmm = vmm_out(ur, ocb);
                vcvtps2dq(vmm, vmm);
            }
        }
    }

    /* write out register to output_addr */
    for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
        const bool mask_flag = last_oc_block && ocb == jcp.nb_oc_blocking - 1;
        for (int ur = 0; ur < ur_w; ur++) {
            int aux_dst_off = jcp.typesize_out
                    * (ur * jcp.ngroups * jcp.oc_without_padding
                            + ocb * jcp.oc_block);
            auto addr = EVEX_compress_addr(reg_dst, aux_dst_off);

            const Vmm vmm = vmm_out(ur, ocb);
            const Vmm r_vmm = mask_flag ? vmm | ktail_mask : vmm;
            switch (jcp.dst_dt) {
                case data_type::f32:
                case data_type::s32: vmovups(addr, r_vmm); break;
                case data_type::s8: vpmovsdb(addr, r_vmm); break;
                case data_type::u8: vpmovusdb(addr, r_vmm); break;
                default: assert(!"unknown dst_dt");
            }
        }
    }
}

template <typename Vmm>
void jit_avx512_core_x8s8s32x_deconv_fwd_kernel<Vmm>::icb_loop(
        int ur_w, int l_overflow, int r_overflow, bool is_last_sp_block) {

    int shift_src_icb = jcp.typesize_in * jcp.ic_block;
    const size_t shift_filt_icb = (size_t)jcp.typesize_in * jcp.kd * jcp.kh
            * jcp.kw * jcp.ic_block * jcp.oc_block;

    prepare_output(ur_w);

    Label skip_icb_loop, icb_loop_label;

    mov(reg_icb, jcp.nb_ic);

    if (zp::should_calculate_deconv_zp_src_pad_str_comp(jcp)) {
        mov(reg_oc_blocks, ptr[param1 + GET_OFF(oc_blocks)]);
        if (jcp.ndims > 3) {
            mov(reg_scratch,
                    qword[param1 + GET_OFF(zp_src_pad_str_compensation)]);
            mov(zp_src_pad_comp_addr, reg_scratch);
        }
    }

    L(icb_loop_label);
    {

        if (jcp.ic_without_padding != jcp.ic) {
            Label common_ker, end_ker;
            cmp(reg_icb, 1);
            jg(common_ker, T_NEAR);

            kh_loop(ur_w, l_overflow, r_overflow,
                    is_last_sp_block ? last_sp_block : last_ic_block);
            jmp(end_ker, T_NEAR);

            L(common_ker);
            kh_loop(ur_w, l_overflow, r_overflow, no_last_block);

            L(end_ker);
        } else {
            kh_loop(ur_w, l_overflow, r_overflow, no_last_block);
        }

        add(reg_src, shift_src_icb);
        safe_add(reg_filt, shift_filt_icb, reg_ker_long_offt);
        dec(reg_icb);
        cmp(reg_icb, 0);
        jg(icb_loop_label, T_NEAR);
    }

    /* come-back pointers */
    sub(reg_src, jcp.nb_ic * shift_src_icb);
    safe_sub(reg_filt, jcp.nb_ic * shift_filt_icb, reg_ker_long_offt);
    L(skip_icb_loop);

    if (jcp.ngroups % jcp.ch_block != 0 || jcp.oc_without_padding != jcp.oc) {
        Label common_store, end_store;
        mov(reg_oc_blocks, ptr[param1 + GET_OFF(oc_blocks)]);
        if (jcp.is_depthwise)
            cmp(reg_oc_blocks, jcp.nb_ch - 1);
        else
            cmp(reg_oc_blocks, jcp.nb_oc - jcp.nb_oc_blocking);
        jne(common_store, T_NEAR);

        store_output(ur_w, true);
        jmp(end_store, T_NEAR);

        L(common_store);
        store_output(ur_w, false);

        L(end_store);

    } else {
        store_output(ur_w, false);
    }
}

template <typename Vmm>
ur_w_blks_params_t
jit_avx512_core_x8s8s32x_deconv_fwd_kernel<Vmm>::get_ur_w_blks_params() {
    const int n_ur_blocks = jcp.ow / jcp.ur_w;

    ur_w_blks_params_t ur_w_blks_params;
    int num_blks_to_process_sp_carefully = 0;
    int idx_last_non_zero_l_overflow_blk = -1;
    int idx_first_non_zero_r_overflow_blk = n_ur_blocks;

    static constexpr int src_pixels_loaded_for_bcast = 4;
    const auto ic_mod = jcp.ic_without_padding % src_pixels_loaded_for_bcast;
    for (int blk_idx = 0; blk_idx < n_ur_blocks; blk_idx++) {
        const int first_blk_dst_elem = blk_idx * jcp.ur_w;
        const int last_dst_blk_elem = first_blk_dst_elem + jcp.ur_w - 1;

        const int last_blk_src_idx = nstl::min(
                jcp.iw - 1, (last_dst_blk_elem + jcp.l_pad) / jcp.stride_w);
        const bool is_out_of_src_pixels_scope
                = ((jcp.iw - 1 - last_blk_src_idx) * jcp.ic_without_padding
                                + ic_mod
                        < src_pixels_loaded_for_bcast);

        const bool process_sp_carefully
                = (ic_mod != 0) && is_out_of_src_pixels_scope;
        const int curr_l_overflow = nstl::max(0,
                ((jcp.kw - 1) * (jcp.dilate_w + 1) - jcp.l_pad
                        - first_blk_dst_elem)
                        / jcp.stride_w);
        const int curr_r_overflow = nstl::max(0,
                (last_dst_blk_elem + jcp.l_pad) / jcp.stride_w - (jcp.iw - 1));

        ur_w_blks_params.blks_params.emplace_back(
                curr_l_overflow, curr_r_overflow, process_sp_carefully);

        num_blks_to_process_sp_carefully
                += static_cast<int>(process_sp_carefully);
        if (curr_l_overflow > 0) idx_last_non_zero_l_overflow_blk = blk_idx;
        if (curr_r_overflow > 0 && idx_first_non_zero_r_overflow_blk > blk_idx)
            idx_first_non_zero_r_overflow_blk = blk_idx;
    }
    idx_first_non_zero_r_overflow_blk
            = nstl::max(idx_first_non_zero_r_overflow_blk,
                    idx_last_non_zero_l_overflow_blk + 1);
    // limit num_r_overflow_blks and num_blks_to_process_last_sp_carefully so that:
    // n_ur_blocks >= num_l_overflow_blks + max(num_r_overflow_blks, num_blks_to_process_last_sp_carefully)
    ur_w_blks_params.num_pre_blks
            = nstl::max(0, idx_last_non_zero_l_overflow_blk + 1);
    const int num_r_overflow_blks = idx_first_non_zero_r_overflow_blk
                    <= idx_last_non_zero_l_overflow_blk
            ? n_ur_blocks - ur_w_blks_params.num_pre_blks
            : n_ur_blocks - idx_first_non_zero_r_overflow_blk;
    num_blks_to_process_sp_carefully
            = ur_w_blks_params.num_pre_blks + num_blks_to_process_sp_carefully
                    < n_ur_blocks
            ? num_blks_to_process_sp_carefully
            : n_ur_blocks - ur_w_blks_params.num_pre_blks;
    ur_w_blks_params.num_post_blks
            = nstl::max(num_r_overflow_blks, num_blks_to_process_sp_carefully);

    return ur_w_blks_params;
}

template <typename Vmm>
void jit_avx512_core_x8s8s32x_deconv_fwd_kernel<Vmm>::generate() {
    preamble();

    if (postops_injector_)
        postops_injector_->push_post_ops_data_on_stack(param1, GET_OFF(post_ops_binary_rhs_arg_vec), reg_src, reg_filt);

    if (zp::should_calculate_deconv_zp_src_pad_str_comp(jcp)) {
        sub(rsp, reserved_stack_size_);
        base_post_ops_data_offset += reserved_stack_size_;
    }

    xor_(reg_scratch, reg_scratch);
    Reg16 _t = reg_scratch.cvt16();
    mov(_t, 0x1);
    vpbroadcastw(vmm_one, _t);

    if (jcp.ngroups % jcp.ch_block != 0 || jcp.oc_without_padding != jcp.oc) {
        int tail_size = jcp.is_depthwise
                ? jcp.ngroups % jcp.ch_block
                : jcp.oc_without_padding % jcp.oc_block;
        int mask = (1 << tail_size) - 1;
        Reg32 regw_tmp = reg_nur_w.cvt32();
        Label skip_tail_mask;
        if (jcp.is_depthwise) {
            kxnorw(ktail_mask, ktail_mask, ktail_mask);
            cmp(dword[param1 + GET_OFF(oc_blocks)], jcp.nb_ch - 1);
            jne(skip_tail_mask, T_NEAR);
        }
        mov(regw_tmp, mask);
        kmovw(ktail_mask, regw_tmp);
        L(skip_tail_mask);
    }

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_filt, ptr[param1 + GET_OFF(filt)]);
    mov(reg_dst, ptr[param1 + GET_OFF(dst)]);

    int dst_shift = jcp.typesize_out * jcp.ur_w * jcp.ngroups
            * jcp.oc_without_padding;
    int src_shift = jcp.typesize_in * (jcp.ur_w / jcp.stride_w) * jcp.ngroups
            * jcp.ic_without_padding;

    const auto ur_w_blks_params = get_ur_w_blks_params();
    const int nur_w = jcp.ow / jcp.ur_w - ur_w_blks_params.num_pre_blks
            - ur_w_blks_params.num_post_blks;

    const auto &blks_params = ur_w_blks_params.blks_params;
    const auto num_pre_blks = ur_w_blks_params.num_pre_blks;
    const auto num_post_blks = ur_w_blks_params.num_post_blks;

    for (int i = 0; i < num_pre_blks; i++) {
        const bool blk_process_carefully = blks_params[i].process_sp_carefully;
        const int blk_l_overflow = blks_params[i].l_overflow;
        const int blk_r_overflow = blks_params[i].r_overflow;

        icb_loop(jcp.ur_w, blk_l_overflow, blk_r_overflow,
                blk_process_carefully);
        add(reg_src, src_shift);
        add(reg_dst, dst_shift);
    }

    if (nur_w > 0) {
        xor_(reg_nur_w, reg_nur_w);
        Label ow_loop_label;
        L(ow_loop_label);
        {
            icb_loop(jcp.ur_w, 0, 0, false);
            add(reg_src, src_shift);
            add(reg_dst, dst_shift);
            inc(reg_nur_w);
            cmp(reg_nur_w, nur_w);
            jl(ow_loop_label, T_NEAR);
        }
    }

    if (num_post_blks > 0) {
        const auto blks_params_size = blks_params.size();
        const auto start_blk_idx = blks_params_size - num_post_blks;
        for (size_t i = start_blk_idx; i < blks_params_size; i++) {
            const bool blk_process_carefully
                    = blks_params[i].process_sp_carefully;
            const int blk_l_overflow = blks_params[i].l_overflow;
            const int blk_r_overflow = blks_params[i].r_overflow;

            icb_loop(jcp.ur_w, blk_l_overflow, blk_r_overflow,
                    blk_process_carefully);
            add(reg_src, src_shift);
            add(reg_dst, dst_shift);
        }
    }

    if (jcp.ur_w_tail != 0) {
        // l_overflow - no. of spatial elements of weights standing out of src spatial
        //              when computing the left-most (in w dim) output pixel
        int l_overflow = 0;
        if (jcp.ur_w == jcp.ow)
            l_overflow = max(0,
                    ((jcp.kw - 1) * (jcp.dilate_w + 1) - jcp.l_pad)
                            / jcp.stride_w);
        // r_overflow - no/ of spatial elements of weights standing out of src spatial
        //              when computing the right-most (in w dim) output pixel
        const int r_overflow = max(0,
                ((jcp.kw - 1) * (jcp.dilate_w + 1) - max(0, jcp.r_pad))
                        / jcp.stride_w);

        icb_loop(jcp.ur_w_tail, l_overflow, r_overflow, true);
    }

    if (zp::should_calculate_deconv_zp_src_pad_str_comp(jcp)) {
        add(rsp, reserved_stack_size_);
        base_post_ops_data_offset -= reserved_stack_size_;
    }

    if (postops_injector_)
        postops_injector_->reset_stack_pointer();

    postamble();

    if (jcp.with_eltwise) postops_injector_->prepare_table();
}

status_t jit_avx512_core_x8s8s32x_deconvolution_fwd_t::execute_forward_1d(
        const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    const auto weights = CTX_IN_MEM(const int8_t *, DNNL_ARG_WEIGHTS);
    const auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);
    DEFINE_ZERO_POINTS_BUFFER(zp_src, DNNL_ARG_SRC);
    DEFINE_ZERO_POINTS_BUFFER(zp_dst, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const size_t dst_dt_size = types::data_type_size(dst_d.data_type());
    const auto &jcp = pd()->jcp_;

    auto scratchpad = ctx.get_scratchpad_grantor();
    int32_t *zp_src_comp_scratch = scratchpad.get<int32_t>(key_deconv_zp);

    if (zp::should_calculate_deconv_zp_src_pad_str_comp(jcp))
        zp::compute_deconv_zp_pad_str_comp_ker(jcp, pd()->with_groups(),
                weights_d, weights, zp_src, zp_src_comp_scratch,
                zp_src_pad_comp_kernel_.get());

    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(jcp.post_ops, ctx);

    const int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    const int nb_groups = jcp.nb_ch;

    const float *oscales = pd()->attr()->output_scales_.scales_;
    if (jcp.signed_input && jcp.ver != ver_vnni) {
        auto local_scales = ctx.get_scratchpad_grantor().template get<float>(
                key_conv_adjusted_scales);
        size_t count = pd()->attr()->output_scales_.count_;
        float factor = 1.f / pd()->jcp_.wei_adj_scale;
        if (count == 1) {
            utils::array_set(local_scales, oscales[0] * factor, 16);
        } else {
            for (size_t c = 0; c < count; c++)
                local_scales[c] = oscales[c] * factor;
        }
        oscales = local_scales;
    }
    const size_t offset = weights_d.size() - weights_d.additional_buffer_size();
    auto w = const_cast<int8_t *>(weights);
    int32_t *compensation = (jcp.signed_input)
            ? reinterpret_cast<int32_t *>(&w[offset])
            : nullptr;
    const int32_t *zp_compensation = jcp.src_zero_point
            ? get_src_zp_comp_from_wei(
                    weights, weights_d, jcp.signed_input, jcp.ngroups, jcp.oc)
            : nullptr;

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        int work_amount = jcp.mb * nb_groups * oc_chunks;
        balance211(work_amount, nthr, ithr, start, end);

        auto p = jit_deconv_call_s();

        int n {0}, g {0}, occ {0};
        if (jcp.loop_order == loop_ngc)
            nd_iterator_init(start, n, jcp.mb, g, nb_groups, occ, oc_chunks);
        else if (jcp.loop_order == loop_cgn)
            nd_iterator_init(start, occ, oc_chunks, g, nb_groups, n, jcp.mb);
        else
            assert(!"unsupported loop order");
        while (start < end) {

            int ocb = occ * jcp.nb_oc_blocking;
            int g_oc = (g * jcp.ch_block * jcp.nb_oc + ocb) * jcp.oc_block;
            int g_ic = g * jcp.ch_block * jcp.ic;

            p.dst = dst + dst_dt_size * dst_d.blk_off(n, g_oc);
            p.src = src + src_d.blk_off(n, g_ic);
            p.filt = weights + wht_blk_off(weights_d, g, ocb, 0);
            p.bias = jcp.with_bias
                    ? bias + (bias_d.blk_off(g_oc) * jcp.typesize_bia)
                    : nullptr;
            p.compensation = (jcp.signed_input) ? compensation + g_oc : nullptr;
            p.scales = &oscales[jcp.is_oc_scale * g_oc];
            p.t_overflow = 0;
            p.b_overflow = 0;
            p.kh_padding = jcp.kh;
            p.oc_blocks = jcp.is_depthwise ? g : ocb;
            p.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec.data();
            p.oc_l_off = g_oc;
            p.oc_off = g_oc * sizeof(float);
            p.zp_compensation
                    = jcp.src_zero_point ? zp_compensation + g_oc : nullptr;
            p.zp_src_pad_str_compensation
                    = jcp.src_zero_point ? zp_src_comp_scratch + g_oc : nullptr;
            p.src_zero_point = zp_src;
            p.dst_zero_point = zp_dst;
            (*kernel_)(&p);

            ++start;
            if (jcp.loop_order == loop_ngc)
                nd_iterator_step(n, jcp.mb, g, nb_groups, occ, oc_chunks);
            else if (jcp.loop_order == loop_cgn)
                nd_iterator_step(occ, oc_chunks, g, nb_groups, n, jcp.mb);
            else
                assert(!"unsupported loop order");
        }
    });
    return status::success;
}

status_t jit_avx512_core_x8s8s32x_deconvolution_fwd_t::execute_forward_2d(
        const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    const auto weights = CTX_IN_MEM(const int8_t *, DNNL_ARG_WEIGHTS);
    const auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);
    DEFINE_ZERO_POINTS_BUFFER(zp_src, DNNL_ARG_SRC);
    DEFINE_ZERO_POINTS_BUFFER(zp_dst, DNNL_ARG_DST);

    const auto &jcp = pd()->jcp_;

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(jcp.post_ops, ctx);

    const size_t dst_dt_size = types::data_type_size(dst_d.data_type());

    auto scratchpad = ctx.get_scratchpad_grantor();
    int32_t *zp_src_comp_scratch = scratchpad.get<int32_t>(key_deconv_zp);

    if (zp::should_calculate_deconv_zp_src_pad_str_comp(jcp))
        zp::compute_deconv_zp_pad_str_comp_ker(jcp, pd()->with_groups(),
                weights_d, weights, zp_src, zp_src_comp_scratch,
                zp_src_pad_comp_kernel_.get());

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int nb_groups = jcp.nb_ch;

    size_t src_h_stride = src_d.blk_off(0, 0, 1);
    size_t dst_h_stride = dst_d.blk_off(0, 0, 1);
    size_t wht_kh_stride = wht_blk_off(weights_d, 0, 0, 0, 1);

    const float *oscales = pd()->attr()->output_scales_.scales_;
    if (jcp.signed_input && jcp.ver != ver_vnni) {
        auto local_scales = ctx.get_scratchpad_grantor().template get<float>(
                key_conv_adjusted_scales);
        size_t count = pd()->attr()->output_scales_.count_;
        float factor = 1.f / pd()->jcp_.wei_adj_scale;
        if (count == 1) {
            utils::array_set(local_scales, oscales[0] * factor, 16);
        } else {
            for (size_t c = 0; c < count; c++)
                local_scales[c] = oscales[c] * factor;
        }
        oscales = local_scales;
    }
    const size_t offset = weights_d.size() - weights_d.additional_buffer_size();
    auto w = const_cast<int8_t *>(weights);
    int32_t *compensation = (jcp.signed_input)
            ? reinterpret_cast<int32_t *>(&w[offset])
            : nullptr;
    const int32_t *zp_compensation = jcp.src_zero_point
            ? get_src_zp_comp_from_wei(
                    weights, weights_d, jcp.signed_input, jcp.ngroups, jcp.oc)
            : nullptr;

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        int work_amount = jcp.mb * nb_groups * oc_chunks * jcp.oh;
        balance211(work_amount, nthr, ithr, start, end);

        auto p = jit_deconv_call_s();

        /*loop order = cgn*/
        int n {0}, g {0}, occ {0}, oh_s {0};
        if (jcp.loop_order == loop_ngc)
            nd_iterator_init(start, n, jcp.mb, g, nb_groups, occ, oc_chunks,
                    oh_s, jcp.oh);
        else if (jcp.loop_order == loop_cgn)
            nd_iterator_init(start, occ, oc_chunks, g, nb_groups, n, jcp.mb,
                    oh_s, jcp.oh);
        else
            assert(!"unsupported loop order");
        while (start < end) {

            int ocb = occ * jcp.nb_oc_blocking;
            int g_oc = (g * jcp.ch_block * jcp.nb_oc + ocb) * jcp.oc_block;
            int g_ic = g * jcp.ch_block * jcp.ic;
            int work_rem = end - start;
            int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;

            auto dst_w = dst + dst_dt_size * dst_d.blk_off(n, g_oc);
            auto src_w = src + src_d.blk_off(n, g_ic);
            auto wht_w = weights + wht_blk_off(weights_d, g, ocb, 0);
            auto bias_w = jcp.with_bias
                    ? bias + (bias_d.blk_off(g_oc) * jcp.typesize_bia)
                    : nullptr;
            int32_t *compensation_w
                    = (jcp.signed_input) ? compensation + g_oc : nullptr;

            auto scales = &oscales[jcp.is_oc_scale * g_oc];
            for (int oj = oh_s; oj < oh_e; oj++) {
                int ih_max = 0, kh_lo = 0, kh_len = 0;
                if (jcp.dilate_h != 0 && jcp.stride_h == 1) {
                    /* dilation */
                    int dilate_h = jcp.dilate_h + 1;
                    // Note: use div_up to account for "holes" in filter
                    int o_t_overflow = div_up(
                            max(0, (jcp.kh - 1) * dilate_h - oj - jcp.t_pad),
                            dilate_h);
                    int o_b_overflow
                            = div_up(max(0,
                                             (jcp.kh - 1) * dilate_h + 1
                                                     - jcp.oh + oj - jcp.b_pad),
                                    dilate_h);
                    kh_len = jcp.kh - o_t_overflow - o_b_overflow;
                    kh_lo = o_b_overflow;
                    ih_max = oj + jcp.t_pad - o_b_overflow * dilate_h;
                } else {
                    int o_t_overflow = max(
                            0, (jcp.kh - (oj + 1 + jcp.t_pad)) / jcp.stride_h);
                    int o_b_overflow = max(0,
                            ((oj + jcp.kh) - (jcp.oh + jcp.b_pad))
                                    / jcp.stride_h);
                    int overflow_kh_hi = jcp.kh - 1
                            - modulo(jcp.oh + jcp.b_pad - (oj + 1),
                                    jcp.stride_h);
                    int overflow_kh_lo = (oj + jcp.t_pad) % jcp.stride_h;

                    kh_len = (overflow_kh_hi - overflow_kh_lo) / jcp.stride_h
                            + 1 - o_t_overflow - o_b_overflow;
                    kh_lo = overflow_kh_lo + o_b_overflow * jcp.stride_h;
                    ih_max = (oj + jcp.t_pad - kh_lo) / jcp.stride_h;
                }

                int wei_stride = (!jcp.signed_input && !jcp.src_zero_point)
                        ? kh_lo * wht_kh_stride
                        : 0;
                p.src = src_w + ih_max * src_h_stride;
                p.dst = dst_w + dst_dt_size * oj * dst_h_stride;
                p.filt = wht_w + wei_stride;
                p.bias = bias_w;
                p.compensation = compensation_w;
                p.t_overflow = jcp.dilate_h > 0
                        ? jcp.kh - kh_len - kh_lo
                        : max(0,
                                jcp.kh
                                        - (kh_lo
                                                + max(0, kh_len - 1)
                                                        * jcp.stride_h
                                                + 1));
                p.b_overflow = kh_lo;
                p.kh_padding = kh_len;
                p.scales = scales;
                p.oc_blocks = jcp.is_depthwise ? g : ocb;
                p.post_ops_binary_rhs_arg_vec
                        = post_ops_binary_rhs_arg_vec.data();
                p.oc_l_off = g_oc;
                p.oc_off = g_oc * sizeof(float);
                p.zp_compensation
                        = jcp.src_zero_point ? zp_compensation + g_oc : nullptr;
                p.zp_src_pad_str_compensation = jcp.src_zero_point
                        ? zp_src_comp_scratch + g_oc
                        : nullptr;
                p.src_zero_point = zp_src;
                p.dst_zero_point = zp_dst;

                (*kernel_)(&p);
            }
            if (jcp.loop_order == loop_ngc)
                nd_iterator_jump(start, end, n, jcp.mb, g, nb_groups, occ,
                        oc_chunks, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_cgn)
                nd_iterator_jump(start, end, occ, oc_chunks, g, nb_groups, n,
                        jcp.mb, oh_s, jcp.oh);
            else
                assert(!"unsupported loop order");
        }
    });
    return status::success;
}

status_t jit_avx512_core_x8s8s32x_deconvolution_fwd_t::execute_forward_3d(
        const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    const auto weights = CTX_IN_MEM(const int8_t *, DNNL_ARG_WEIGHTS);
    const auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);
    DEFINE_ZERO_POINTS_BUFFER(zp_src, DNNL_ARG_SRC);
    DEFINE_ZERO_POINTS_BUFFER(zp_dst, DNNL_ARG_DST);

    const auto &jcp = pd()->jcp_;

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(jcp.post_ops, ctx);

    const size_t dst_dt_size = types::data_type_size(dst_d.data_type());

    auto scratchpad = ctx.get_scratchpad_grantor();
    int32_t *zp_src_comp_scratch = scratchpad.get<int32_t>(key_deconv_zp);

    if (zp::should_calculate_deconv_zp_src_pad_str_comp(jcp))
        zp::compute_deconv_zp_pad_str_comp_ker(jcp, pd()->with_groups(),
                weights_d, weights, zp_src, zp_src_comp_scratch,
                zp_src_pad_comp_kernel_.get());

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int nb_groups = jcp.nb_ch;

    size_t src_d_stride = src_d.blk_off(0, 0, 1);
    size_t src_h_stride = src_d.blk_off(0, 0, 0, 1);
    size_t dst_d_stride = dst_d.blk_off(0, 0, 1);
    size_t dst_h_stride = dst_d.blk_off(0, 0, 0, 1);
    size_t wht_kd_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
    size_t wht_kh_stride = wht_blk_off(weights_d, 0, 0, 0, 0, 1);

    const float *oscales = pd()->attr()->output_scales_.scales_;
    if (jcp.signed_input && jcp.ver != ver_vnni) {
        auto local_scales = ctx.get_scratchpad_grantor().template get<float>(
                key_conv_adjusted_scales);
        size_t count = pd()->attr()->output_scales_.count_;
        float factor = 1.f / pd()->jcp_.wei_adj_scale;
        if (count == 1) {
            utils::array_set(local_scales, oscales[0] * factor, 16);
        } else {
            for (size_t c = 0; c < count; c++)
                local_scales[c] = oscales[c] * factor;
        }
        oscales = local_scales;
    }
    size_t offset = weights_d.size() - weights_d.additional_buffer_size();
    auto w = const_cast<int8_t *>(weights);
    int32_t *compensation = (jcp.signed_input)
            ? reinterpret_cast<int32_t *>(&w[offset])
            : nullptr;
    const int32_t *zp_compensation = jcp.src_zero_point
            ? get_src_zp_comp_from_wei(
                    weights, weights_d, jcp.signed_input, jcp.ngroups, jcp.oc)
            : nullptr;

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        int work_amount = jcp.mb * nb_groups * oc_chunks * jcp.od * jcp.oh;
        balance211(work_amount, nthr, ithr, start, end);

        auto p = jit_deconv_call_s();

        /*loop order = cgn*/
        int n {0}, g {0}, occ {0}, od_s {0}, oh_s {0};
        if (jcp.loop_order == loop_ngc)
            nd_iterator_init(start, n, jcp.mb, g, nb_groups, occ, oc_chunks,
                    od_s, jcp.od, oh_s, jcp.oh);
        else if (jcp.loop_order == loop_cgn)
            nd_iterator_init(start, occ, oc_chunks, g, nb_groups, n, jcp.mb,
                    od_s, jcp.od, oh_s, jcp.oh);
        else
            assert(!"unsupported loop order");
        while (start < end) {

            int ocb = occ * jcp.nb_oc_blocking;
            int g_oc = (g * jcp.ch_block * jcp.nb_oc + ocb) * jcp.oc_block;
            int g_ic = g * jcp.ch_block * jcp.ic;
            int work_rem = end - start;
            int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
            int input_d_s = 0, kd_len = 0, kd_lo = 0;
            int d_t_overflow, d_back_overflow;

            if (jcp.dilate_d != 0 && jcp.stride_d == 1) {
                /* dilation */
                int dilate_d = jcp.dilate_d + 1;
                // Note: use div_up to account for "holes" in filter
                d_t_overflow = div_up(
                        max(0, (jcp.kd - 1) * dilate_d - od_s - jcp.f_pad),
                        dilate_d);
                d_back_overflow
                        = div_up(max(0,
                                         (jcp.kd - 1) * dilate_d + 1 - jcp.od
                                                 + od_s - jcp.back_pad),
                                dilate_d);
                kd_len = jcp.kd - d_t_overflow - d_back_overflow;
                kd_lo = d_back_overflow;
                input_d_s = od_s + jcp.f_pad - d_back_overflow * dilate_d;
            } else {
                int d_t_overflow = max(
                        0, (jcp.kd - (od_s + 1 + jcp.f_pad)) / jcp.stride_d);
                int d_back_overflow = max(0,
                        ((od_s + jcp.kd) - (jcp.od + jcp.back_pad))
                                / jcp.stride_d);
                int overflow_kd_hi = jcp.kd - 1
                        - modulo(jcp.od + jcp.back_pad - (od_s + 1),
                                jcp.stride_d);
                int overflow_kd_lo = (od_s + jcp.f_pad) % jcp.stride_d;

                kd_len = (overflow_kd_hi - overflow_kd_lo) / jcp.stride_d + 1
                        - d_t_overflow - d_back_overflow;
                kd_lo = overflow_kd_lo + d_back_overflow * jcp.stride_d;
                input_d_s = (od_s + jcp.f_pad - kd_lo) / jcp.stride_d;
            }

            auto dst_w = dst
                    + dst_dt_size
                            * (dst_d.blk_off(n, g_oc) + od_s * dst_d_stride);
            auto src_w
                    = src + src_d.blk_off(n, g_ic) + input_d_s * src_d_stride;
            auto wht_w = weights + wht_blk_off(weights_d, g, ocb, 0)
                    + ((jcp.signed_input || jcp.src_zero_point) ? 0 : kd_lo)
                            * wht_kd_stride;
            auto bias_w = jcp.with_bias
                    ? bias + (bias_d.blk_off(g_oc) * jcp.typesize_bia)
                    : nullptr;
            int32_t *compensation_w
                    = (jcp.signed_input) ? compensation + g_oc : nullptr;

            auto scales = &oscales[jcp.is_oc_scale * g_oc];

            for (int oj = oh_s; oj < oh_e; oj++) {
                int ih_max = 0, kh_lo = 0, kh_len = 0;
                if (jcp.dilate_h != 0 && jcp.stride_h == 1) {
                    /* dilation */
                    int dilate_h = jcp.dilate_h + 1;
                    // Note: use div_up to account for "holes" in filter
                    int o_t_overflow = div_up(
                            max(0, (jcp.kh - 1) * dilate_h - oj - jcp.t_pad),
                            dilate_h);
                    int o_b_overflow
                            = div_up(max(0,
                                             (jcp.kh - 1) * dilate_h + 1
                                                     - jcp.oh + oj - jcp.b_pad),
                                    dilate_h);
                    kh_len = jcp.kh - o_t_overflow - o_b_overflow;
                    kh_lo = o_b_overflow;
                    ih_max = oj + jcp.t_pad - o_b_overflow * dilate_h;
                } else {
                    int o_t_overflow = max(
                            0, (jcp.kh - (oj + 1 + jcp.t_pad)) / jcp.stride_h);
                    int o_b_overflow = max(0,
                            ((oj + jcp.kh) - (jcp.oh + jcp.b_pad))
                                    / jcp.stride_h);
                    int overflow_kh_hi = jcp.kh - 1
                            - modulo(jcp.oh + jcp.b_pad - (oj + 1),
                                    jcp.stride_h);
                    int overflow_kh_lo = (oj + jcp.t_pad) % jcp.stride_h;

                    kh_len = (overflow_kh_hi - overflow_kh_lo) / jcp.stride_h
                            + 1 - o_t_overflow - o_b_overflow;
                    kh_lo = overflow_kh_lo + o_b_overflow * jcp.stride_h;
                    ih_max = (oj + jcp.t_pad - kh_lo) / jcp.stride_h;
                }

                int wei_stride = (!jcp.signed_input && !jcp.src_zero_point)
                        ? kh_lo * wht_kh_stride
                        : 0;
                p.src = src_w + ih_max * src_h_stride;
                p.dst = dst_w + dst_dt_size * oj * dst_h_stride;
                p.filt = wht_w + wei_stride;
                p.bias = bias_w;
                p.compensation = compensation_w;
                /* Note: Currently this kernel doesn't support dilations and
                strides together */
                p.t_overflow = jcp.dilate_h > 0
                        ? jcp.kh - kh_len - kh_lo
                        : max(0,
                                jcp.kh
                                        - (kh_lo
                                                + max(0, kh_len - 1)
                                                        * jcp.stride_h
                                                + 1));
                p.b_overflow = kh_lo;
                p.f_overflow = jcp.dilate_d > 0
                        ? jcp.kd - kd_len - kd_lo
                        : max(0,
                                jcp.kd
                                        - (kd_lo
                                                + max(0, kd_len - 1)
                                                        * jcp.stride_d
                                                + 1));
                p.back_overflow = kd_lo;
                p.kh_padding = kh_len;
                p.kd_padding = kd_len;
                p.scales = scales;
                p.oc_blocks = jcp.is_depthwise ? g : ocb;
                p.post_ops_binary_rhs_arg_vec
                        = post_ops_binary_rhs_arg_vec.data();
                p.oc_l_off = g_oc;
                p.oc_off = g_oc * sizeof(float);
                p.zp_compensation
                        = jcp.src_zero_point ? zp_compensation + g_oc : nullptr;
                p.zp_src_pad_str_compensation = jcp.src_zero_point
                        ? zp_src_comp_scratch + g_oc
                        : nullptr;
                p.src_zero_point = zp_src;
                p.dst_zero_point = zp_dst;
                (*kernel_)(&p);
            }
            if (jcp.loop_order == loop_ngc)
                nd_iterator_jump(start, end, n, jcp.mb, g, nb_groups, occ,
                        oc_chunks, od_s, jcp.od, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_cgn)
                nd_iterator_jump(start, end, occ, oc_chunks, g, nb_groups, n,
                        jcp.mb, od_s, jcp.od, oh_s, jcp.oh);
            else
                assert(!"unsupported loop order");
        }
    });
    return status::success;
}

template struct jit_avx512_core_x8s8s32x_deconv_fwd_kernel<Xbyak::Zmm>;
template struct jit_avx512_core_x8s8s32x_deconv_fwd_kernel<Xbyak::Ymm>;
template struct jit_avx512_core_x8s8s32x_deconv_fwd_kernel<Xbyak::Xmm>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
