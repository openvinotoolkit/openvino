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

#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/zero_point_utils.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_uni_deconv_zp_pad_str_kernel.hpp"
#include "cpu/x64/jit_uni_x8s8s32x_deconvolution.hpp"

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

template <cpu_isa_t isa>
status_t jit_uni_x8s8s32x_deconv_fwd_kernel<isa>::init_conf(
        jit_conv_conf_t &jcp, const deconvolution_desc_t &cd,
        memory_desc_t &src_md, memory_desc_t &weights_md, memory_desc_t &dst_md,
        const bool with_bias, memory_desc_t &bias_md, primitive_attr_t &attr,
        int nthreads) {
    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper bias_d(&bias_md);

    if (!(mayiuse(isa)
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
    const bool is_avx2 = isa == avx2;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.id = is_3d ? src_d.dims()[2] : 1;
    jcp.oc_without_padding = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = src_d.dims()[1] / jcp.ngroups;
    jcp.is_depthwise = true && with_groups
            && utils::everyone_is(
                    1, jcp.ic_without_padding, jcp.oc_without_padding);
    jcp.ver = mayiuse(avx2_vnni) ? ver_vnni : ver_unused;

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
        if (jcp.ic_block == 8 || jcp.ch_block == 8) {
            if (is_1d) {
                wei_tag = with_groups ? jcp.is_depthwise ? Goiw8g : gOIw2i8o4i
                                      : OIw2i8o4i;
            } else if (is_2d) {
                wei_tag = with_groups ? jcp.is_depthwise ? Goihw8g : gOIhw2i8o4i
                                      : OIhw2i8o4i;
            } else {
                wei_tag = with_groups ? gOIdhw2i8o4i : OIdhw2i8o4i;
            }
        } else {
            if (is_avx2) {
                assert(with_groups && jcp.ic_block == 4);
                wei_tag = is_3d ? gOIdhw4o4i : is_2d ? gOIhw4o4i : gOIw4o4i;
            } else {
                if (is_1d) {
                    wei_tag = with_groups ? jcp.is_depthwise ? Goiw4g : gOIw4o4i
                                          : OIw4o4i;
                } else if (is_2d) {
                    wei_tag = with_groups
                            ? jcp.is_depthwise ? Goihw4g : gOIhw4o4i
                            : OIhw4o4i;
                } else {
                    wei_tag = with_groups ? gOIdhw4o4i : OIdhw4o4i;
                }
            }
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
        jcp.ch_block = is_avx2 ? 8 : 4;
        jcp.oc_block = 1;
        jcp.ic_block = 1;
    } else {
        jcp.ch_block = 1;
        jcp.oc_block = is_avx2 ? 8 : 4;
        jcp.ic_block = is_avx2 ? 8 : 4;

        if (jcp.ngroups == 1) {
            jcp.oc = utils::rnd_up(jcp.oc_without_padding, jcp.oc_block);
            jcp.ic = utils::rnd_up(jcp.ic_without_padding, jcp.ic_block);
        } else if (jcp.ngroups != 1
                && ((jcp.ic % jcp.ic_block != 0)
                        || (jcp.oc % jcp.oc_block != 0))) {
            /* For grouped convolution, oneDNN doesn't support padding.
             * When channel per group is not multiple of 8 in avx2:
             * - Use Xmm when channels per groups is multiple of 4.
             * - Otherwise return unimplemented */
            jcp.oc_block = jcp.ic_block = 4;
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

    const int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    const int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    const int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.iw, jcp.ow, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.ih, jcp.oh, jcp.stride_h, ext_kh);
    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.id, jcp.od, jcp.stride_d, ext_kd);
    const bool kernel_outside_src = false || ext_kw <= jcp.l_pad
            || ext_kw <= jcp.r_pad || ext_kh <= jcp.t_pad || ext_kh <= jcp.b_pad
            || ext_kd <= jcp.f_pad || ext_kd <= jcp.back_pad;
    if (kernel_outside_src) return status::unimplemented;

    CHECK(attr.set_default_formats(&dst_md));
    if (!post_ops_ok(jcp, dst_d, attr)) return status::unimplemented;

    const auto &p = attr.post_ops_;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) jcp.eltwise = p.entry_[eltwise_ind].eltwise;

    const int binary_ind = p.find(primitive_kind::binary);
    jcp.with_binary = binary_ind != -1;

    const int sum_ind = p.find(primitive_kind::sum);
    jcp.with_sum = sum_ind != -1;

    const int depthwise_ind = p.find(primitive_kind::depthwise);
    jcp.with_depthwise = depthwise_ind != -1;

    const int quantization_ind = p.find(primitive_kind::quantization);
    jcp.with_quantization = quantization_ind != -1;

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    jcp.post_ops = p;

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
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    /* kernel blocking params */
    const int regs = (jcp.ver == ver_vnni ? 14 : 12);

    jcp.nb_ch_blocking = 1;
    jcp.nb_oc_blocking = nstl::min(4, jcp.nb_oc);
    for (; jcp.nb_oc_blocking > 1; jcp.nb_oc_blocking--)
        if (jcp.nb_oc % jcp.nb_oc_blocking == 0
                && jcp.l_pad <= regs / (jcp.nb_oc_blocking + 1))
            break;

    jcp.ur_w = regs / (jcp.nb_oc_blocking + 1);
    const int l_overflow = max(
            0, ((jcp.kw - 1) * (jcp.dilate_w + 1) - jcp.l_pad) / jcp.stride_w);

    if (jcp.ow < jcp.ur_w) {
        jcp.ur_w = jcp.ow;
        jcp.ur_w_tail = 0;
    } else {
        for (; jcp.ur_w >= 1; jcp.ur_w--) {
            /* ur_w should be multiple of stride_w in order
               to simplify logic for get_ow_start and get_ow_end */
            const bool is_multiple_of_stride = jcp.ur_w % jcp.stride_w == 0;

            /* boundary conditions:
               These conditions ensure all elements close to boundary
               are computed in a single call of compute loop */
            const bool left_boundary_covered
                    = jcp.ur_w >= l_overflow * jcp.stride_w;
            jcp.ur_w_tail = jcp.ow % jcp.ur_w;
            const int r_overflow_no_tail = max(0,
                    ((jcp.kw - 1) * (jcp.dilate_w + 1) - max(0, jcp.r_pad)
                            - jcp.ur_w_tail)
                            / jcp.stride_w);
            const bool right_boundary_covered
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

template <cpu_isa_t isa>
jit_uni_x8s8s32x_deconv_fwd_kernel<isa>::jit_uni_x8s8s32x_deconv_fwd_kernel(
        const jit_conv_conf_t &ajcp, const primitive_attr_t &attr,
        const memory_desc_wrapper &dst_d)
    : kernel_(nullptr) {

    const int ch_block = ajcp.is_depthwise ? ajcp.ch_block : ajcp.ic_block;
    switch (ch_block) {
        case 8:
            if (isa == avx2) {
                kernel_ = utils::make_unique<
                        _jit_avx2_x8s8s32x_deconv_fwd_kernel>(
                        ajcp, attr, dst_d);
                return;
            } else
                assert(!"invalid channel blocking for current ISA");
        case 4:
            kernel_ = utils::make_unique<
                    _jit_uni_x8s8s32x_deconv_fwd_kernel<isa, Xbyak::Xmm>>(
                    ajcp, attr, dst_d);
            return;
        default: assert(!"invalid channel blocking");
    }
}

template <cpu_isa_t isa>
jit_uni_x8s8s32x_deconv_fwd_kernel<isa>::~jit_uni_x8s8s32x_deconv_fwd_kernel()
        = default;

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_deconv_fwd_kernel<isa>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp,
        const primitive_attr_t &attr) {
    if (jcp.signed_input && jcp.ver != ver_vnni) {
        dim_t count = nstl::max<dim_t>(attr.output_scales_.count_, 8);
        scratchpad.book<float>(key_conv_adjusted_scales, count);
    }

    if (zp::should_calculate_deconv_zp_src_pad_str_comp(jcp)) {
        const dim_t zp_pad_comp_size = jcp.oc_without_padding * jcp.ngroups
                * jcp.kd * jcp.kh * jcp.kw;
        scratchpad.book<int32_t>(key_deconv_zp, zp_pad_comp_size);
    }
}

template <cpu_isa_t isa>
bool jit_uni_x8s8s32x_deconv_fwd_kernel<isa>::post_ops_ok(jit_conv_conf_t &jcp,
        const memory_desc_wrapper &dst_d, const primitive_attr_t &attr) {
    using namespace injector;

    return injector::post_ops_ok(post_ops_ok_args_t(isa, {sum, eltwise, binary, depthwise, quantization},
            attr.post_ops_, &dst_d, false /*sum_at_pos_0_only*/,
            false /*sum_requires_scale_one*/, false /*sum_requires_zp_zero*/,
            {broadcasting_strategy_t::per_oc,
                    broadcasting_strategy_t::scalar}));
}

template <cpu_isa_t isa, typename Vmm>
_jit_uni_x8s8s32x_deconv_fwd_kernel<isa,
        Vmm>::_jit_uni_x8s8s32x_deconv_fwd_kernel(const jit_conv_conf_t &ajcp,
        const primitive_attr_t &attr, const memory_desc_wrapper &dst_d)
    : jit_generator(nullptr, MAX_CODE_SIZE, true, isa)
    , jcp_(ajcp)
    , postops_injector_(nullptr) {

    if (jcp_.with_eltwise || jcp_.with_binary || jcp_.with_sum || jcp_.with_depthwise || jcp_.with_quantization) {
        const std::size_t tail_size = get_tail_size();

        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = true;
        static constexpr bool use_exact_tail_scalar_bcast = false;
        static constexpr size_t vmm_helper_idx = 15;

        const binary_injector::rhs_arg_static_params_t rhs_sp {vmm_helper_idx,
                this->rdx, this->r10, preserve_gpr, preserve_vmm,
                GET_OFF(post_ops_binary_rhs_arg_vec), dst_d, tail_size,
                Xbyak::Opmask(2), use_exact_tail_scalar_bcast};
        const binary_injector::static_params_t bsp {this->param1_, rhs_sp};

        const quantization_injector::static_params_t qsp {vmm_d_weights.getIdx(), vmm_d_bias.getIdx(), reg_d_weights, reg_d_bias};

        postops_injector_ = utils::make_unique<
                injector::jit_uni_postops_injector_t<isa, Vmm>>(
                this, jcp_.post_ops, bsp, qsp);
    }
}

template <cpu_isa_t isa, typename Vmm>
_jit_uni_x8s8s32x_deconv_fwd_kernel<isa,
        Vmm>::~_jit_uni_x8s8s32x_deconv_fwd_kernel()
        = default;

template <cpu_isa_t isa, typename Vmm>
Vmm _jit_uni_x8s8s32x_deconv_fwd_kernel<isa, Vmm>::vmm_out(
        int i_ur, int i_oc) const {
    const int idx = i_ur * jcp_.nb_oc_blocking + i_oc;
    assert(idx < KER_MAX_REG_IDX);
    /* remap the reg indices to avoid using xmm0 in eltwise injector */
    return Vmm(15 - idx);
}

template <cpu_isa_t isa, typename Vmm>
Vmm _jit_uni_x8s8s32x_deconv_fwd_kernel<isa, Vmm>::vmm_inp(
        int i_ic, int nb_x_blocking) const {
    const int idx = i_ic + nb_x_blocking * jcp_.ur_w;
    assert(idx < KER_MAX_REG_IDX);
    return Vmm(15 - idx);
}

template <cpu_isa_t isa, typename Vmm>
Vmm _jit_uni_x8s8s32x_deconv_fwd_kernel<isa, Vmm>::vmm_bias_alpha() const {
    return Vmm(15 - jcp_.nb_oc_blocking * jcp_.ur_w);
}

template <cpu_isa_t isa, typename Vmm>
Xmm _jit_uni_x8s8s32x_deconv_fwd_kernel<isa, Vmm>::xmm_bias_alpha() const {
    return Xmm(vmm_bias_alpha().getIdx());
}

template <cpu_isa_t isa, typename Vmm>
int _jit_uni_x8s8s32x_deconv_fwd_kernel<isa, Vmm>::get_ow_start(
        int ki, int l_overflow) const noexcept {
    int res = (jcp_.ow - 1 + jcp_.r_pad) % jcp_.stride_w
            + l_overflow * jcp_.stride_w
            - (jcp_.kw - 1 - ki) * (jcp_.dilate_w + 1);
    while (res < 0)
        res += jcp_.stride_w;
    return res;
}

template <cpu_isa_t isa, typename Vmm>
int _jit_uni_x8s8s32x_deconv_fwd_kernel<isa, Vmm>::get_ow_end(
        int ur_w, int ki, int r_overflow) const noexcept {
    if (utils::one_of(ur_w, jcp_.ow, jcp_.ur_w_tail))
        ur_w += nstl::min(0, jcp_.r_pad); // remove negative padding
    int res = (ur_w - 1 + jcp_.l_pad) % jcp_.stride_w
            + r_overflow * jcp_.stride_w - ki * (jcp_.dilate_w + 1);
    while (res < 0)
        res += jcp_.stride_w;
    return ur_w - res;
}

template <cpu_isa_t isa, typename Vmm>
int _jit_uni_x8s8s32x_deconv_fwd_kernel<isa, Vmm>::get_blocking_size() const
        noexcept {
    return jcp_.is_depthwise ? jcp_.ch_block : jcp_.oc_block;
}

template <cpu_isa_t isa, typename Vmm>
int _jit_uni_x8s8s32x_deconv_fwd_kernel<isa, Vmm>::get_tail_size() const
        noexcept {
    return jcp_.is_depthwise ? jcp_.ngroups % jcp_.ch_block
                             : jcp_.oc_without_padding % jcp_.oc_block;
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_deconv_fwd_kernel<isa, Vmm>::compute(
        const Vmm &vreg_acc, const Vmm &vreg_wei, const Vmm &vreg_src) {

    if (jcp_.ver == ver_vnni) {
        vpdpbusd(vreg_acc, vreg_src, vreg_wei, Xbyak::VexEncoding);
    } else if (jcp_.is_depthwise) {
        uni_vmovups(vmm_tmp_, vreg_src);
        uni_vpmulld(vmm_tmp_, vmm_tmp_, vreg_wei);
        uni_vpaddd(vreg_acc, vreg_acc, vmm_tmp_);
    } else {
        uni_vpmaddubsw(vmm_tmp_, vreg_src, vreg_wei);
        uni_vpmaddwd(vmm_tmp_, vmm_tmp_, vmm_one_);
        uni_vpaddd(vreg_acc, vreg_acc, vmm_tmp_);
    }
}

template <cpu_isa_t isa, typename Vmm>
std::function<Vmm()> _jit_uni_x8s8s32x_deconv_fwd_kernel<isa,
        Vmm>::prepare_round_robin_vmm_inp_generator(int ur_w) const noexcept {

    const int start_vmm_idx = vmm_inp(ur_w - 1, jcp_.nb_oc_blocking).getIdx();
    const int end_vmm_idx = vmm_inp(0, jcp_.nb_oc_blocking).getIdx() + 1;
    int current_vmm_idx = start_vmm_idx;

    return [=]() mutable {
        const Vmm vmm {static_cast<int>(current_vmm_idx++)};

        if (current_vmm_idx == end_vmm_idx) current_vmm_idx = start_vmm_idx;

        return vmm;
    };
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_deconv_fwd_kernel<isa, Vmm>::apply_zp_src_pad_str_comp(
        int ur_w, int l_overflow, int r_overflow, bool h_padded) {
    Xbyak::Label end_zp_pad, no_tail;

    // apply once per icb loop, zp src stride paddding compensation calculate as
    // zp_pad_str_compensation = conv(1, weights_s8) * zero_point_source
    cmp(reg_icb_, jcp_.nb_ic);
    jne(end_zp_pad, T_NEAR);

    if (jcp_.ngroups % jcp_.ch_block
            || jcp_.oc_without_padding % jcp_.oc_block) {
        if (jcp_.is_depthwise)
            cmp(reg_oc_blocks_, jcp_.nb_ch - 1);
        else
            cmp(reg_oc_blocks_, jcp_.nb_oc - jcp_.nb_oc_blocking);
        jne(no_tail, T_NEAR);

        append_zp_src_pad_str_comp(
                ur_w, l_overflow, r_overflow, h_padded, true /*last_oc_block*/);
        jmp(end_zp_pad, T_NEAR);
    }

    L(no_tail);
    append_zp_src_pad_str_comp(
            ur_w, l_overflow, r_overflow, h_padded, false /*last_oc_block*/);

    L(end_zp_pad);
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_deconv_fwd_kernel<isa, Vmm>::append_zp_src_pad_str_comp(
        int ur_w, int l_overflow, int r_overflow, bool h_padded,
        bool last_oc_block) {

    const auto &reg_zp_src_pad_comp = reg_scratch_;
    const auto get_next_comp_vmm = prepare_round_robin_vmm_inp_generator(ur_w);
    bool base_comp_addr_loaded = false;

    const auto load_base_zp_src_pad_comp_addr = [&]() {
        if (!base_comp_addr_loaded) {
            if (jcp_.ndims == 5) mov(reg_scratch_preserved_, reg_scratch_);

            if (jcp_.ndims > 3)
                mov(reg_zp_src_pad_comp, zp_src_pad_comp_addr_);
            else
                mov(reg_zp_src_pad_comp,
                        qword[param1_ + GET_OFF(zp_src_pad_str_compensation)]);

            base_comp_addr_loaded = true;
        }
    };

    const auto load_zp_src_pad_comp = [&](const Vmm &zp_pad_comp_vmm,
                                              const Xbyak::Address &comp_addr,
                                              const int ocb) {
        const bool is_tail = last_oc_block && ocb == jcp_.nb_oc_blocking - 1;

        if (is_tail)
            load_data(data_type::s32, zp_pad_comp_vmm, comp_addr,
                    get_tail_size());
        else
            uni_vmovups(zp_pad_comp_vmm, comp_addr);
    };

    const auto get_zp_src_comp_pad_off = [&](int it_kw, int ocb) {
        const auto kw_offset = it_kw * jcp_.oc_without_padding * jcp_.ngroups;
        const auto oc_offset = ocb * jcp_.oc_block;

        return (kw_offset + oc_offset) * sizeof(int32_t);
    };

    for (int it_kw = 0; it_kw < jcp_.kw; ++it_kw) {
        const int ow_start = get_ow_start(it_kw, l_overflow);
        const int ow_end = get_ow_end(ur_w, it_kw, r_overflow);

        for (int ocb = 0; ocb < jcp_.nb_oc_blocking; ocb++) {
            Vmm zp_src_comp_pad_vmm; // will be assigned later
            bool ocb_zp_loaded = false;

            const auto zp_src_comp_pad_off
                    = get_zp_src_comp_pad_off(it_kw, ocb);

            for (int it_ow = 0; it_ow < ur_w; ++it_ow) {

                const bool inside_padded_area = h_padded
                        || !(it_ow >= ow_start && it_ow < ow_end
                                && ((it_ow + jcp_.l_pad - it_kw) % jcp_.stride_w
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

    if (jcp_.ndims > 3) {
        if (!base_comp_addr_loaded) load_base_zp_src_pad_comp_addr();

        const auto kh_offset = jcp_.kw * jcp_.oc_without_padding * jcp_.ngroups
                * sizeof(int32_t);

        add(reg_zp_src_pad_comp, kh_offset);
        mov(zp_src_pad_comp_addr_, reg_zp_src_pad_comp);
    }

    if (jcp_.ndims == 5 && base_comp_addr_loaded)
        mov(reg_scratch_, reg_scratch_preserved_);
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_deconv_fwd_kernel<isa, Vmm>::compute_ker(int ur_w,
        int l_overflow, int r_overflow, ker_block_t last_ic_block_flag,
        bool h_padded) {

    const bool signed_input_or_src_zp
            = (jcp_.signed_input || jcp_.src_zero_point);

    const int ch_block_all = jcp_.ch_block * jcp_.ic_block * jcp_.oc_block;
    const int ur_w_stride = signed_input_or_src_zp ? 1 : jcp_.stride_w;

    const auto src_offset = [=](int oj, int icb, int ki) {
        return jcp_.typesize_in
                * (((oj + jcp_.l_pad - ki * (jcp_.dilate_w + 1))
                           / jcp_.stride_w)
                                * jcp_.ngroups * jcp_.ic_without_padding
                        + icb * 4);
    };

    const auto kernel_offset = [=](int ocb, int icb, int ki) {
        return jcp_.typesize_in
                * ((ocb * jcp_.nb_ic * jcp_.kd * jcp_.kh * jcp_.kw + ki)
                                * ch_block_all
                        + icb * jcp_.oc_block * IC_SUB_STEP);
    };

    for (int ki = 0; ki < jcp_.kw; ki++) {

        const int jj_start = get_ow_start(ki, l_overflow);
        const int jj_end = get_ow_end(ur_w, ki, r_overflow);

        const int _start = (signed_input_or_src_zp) ? 0 : jj_start;
        const int _end = (signed_input_or_src_zp) ? ur_w : jj_end;

        const int tail_size = jcp_.is_depthwise ? jcp_.ngroups % jcp_.ch_block
                                                : jcp_.ic_without_padding % 4;
        const int n_ic_blocks = jcp_.is_depthwise
                ? 1
                : (last_ic_block_flag != no_last_block ? div_up(
                           jcp_.ic_without_padding % jcp_.ic_block, 4)
                                                       : jcp_.ic_block / 4);

        for (int icb1 = 0; icb1 < n_ic_blocks; icb1++) {
            if (h_padded) {
                /* fill padded area with shifted values */
                if (jcp_.signed_input) {
                    const Vmm inp = vmm_inp(0, jcp_.nb_oc_blocking);
                    uni_vpxor(inp, inp, inp);
                    uni_vpsubb(inp, inp, vmm_shift_);
                }
            } else {

                for (int jj = _start; jj < _end; jj += ur_w_stride) {

                    const int aux_src_off = src_offset(jj, icb1, ki);
                    const auto vmm_src = vmm_inp(jj, jcp_.nb_oc_blocking);

                    if (jj >= jj_start && jj < jj_end
                            && ((jj + jcp_.l_pad - ki) % jcp_.stride_w == 0)) {
                        if (jcp_.is_depthwise) {
                            if (tail_size != 0)
                                assert(jcp_.nb_oc_blocking == 1);
                            uni_vpxor(vmm_src, vmm_src, vmm_src);
                            const bool mask_flag
                                    = last_ic_block_flag != no_last_block
                                    && tail_size;
                            load_data(data_type::u8, vmm_src, aux_reg_src_,
                                    aux_src_off,
                                    mask_flag ? tail_size : jcp_.ch_block);
                        } else if ((last_ic_block_flag == last_sp_block)
                                && tail_size != 0 && icb1 == n_ic_blocks - 1) {
                            const auto vmm_inp_tmp = Xmm(vmm_src.getIdx());
                            load_bytes(vmm_inp_tmp, aux_reg_src_, aux_src_off,
                                    tail_size);
                            uni_vpbroadcastd(vmm_src, vmm_inp_tmp);
                        } else {
                            uni_vpbroadcastd(
                                    vmm_src, ptr[aux_reg_src_ + aux_src_off]);
                        }
                        if (jcp_.signed_input)
                            uni_vpsubb(vmm_src, vmm_src, vmm_shift_);
                    } else {
                        /* fill padded area with shifted values */
                        if (jcp_.signed_input) {
                            uni_vpxor(vmm_src, vmm_src, vmm_src);
                            uni_vpsubb(vmm_src, vmm_src, vmm_shift_);
                        }
                    }
                }
            }
            for (int ocb = 0; ocb < jcp_.nb_oc_blocking; ocb++) {
                const int aux_filt_off = kernel_offset(ocb, icb1, ki);

                if (_end - _start > 0) {
                    if (jcp_.is_depthwise) {
                        uni_vpmovsxbd(
                                vmm_wei_, ptr[aux_reg_filt_ + aux_filt_off]);
                    } else
                        uni_vmovups(
                                vmm_wei_, ptr[aux_reg_filt_ + aux_filt_off]);
                }

                for (int jj = _start; jj < _end; jj += ur_w_stride) {

                    const bool inside_padded_area = h_padded
                            || !(jj >= jj_start && jj < jj_end
                                    && ((jj + jcp_.l_pad - ki) % jcp_.stride_w
                                            == 0));
                    const auto vmm_dst = vmm_out(jj, ocb);
                    if (jcp_.signed_input || !inside_padded_area) {
                        const Vmm inp = vmm_inp(
                                h_padded ? 0 : jj, jcp_.nb_oc_blocking);
                        compute(vmm_dst, vmm_wei_, inp);
                    }
                }
            }
        }
    }

    if (zp::should_calculate_deconv_zp_src_pad_str_comp(jcp_))
        apply_zp_src_pad_str_comp(ur_w, l_overflow, r_overflow, h_padded);
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_deconv_fwd_kernel<isa, Vmm>::kh_loop(int ur_w,
        int l_overflow, int r_overflow, ker_block_t last_ic_block_flag) {

    const bool signed_input_or_src_zp
            = (jcp_.signed_input || jcp_.src_zero_point);

    const int ch_block_all = jcp_.ch_block * jcp_.ic_block * jcp_.oc_block;
    const int shift_src_ih = jcp_.typesize_in * (jcp_.dilate_h + 1) * jcp_.iw
            * jcp_.ngroups * jcp_.ic_without_padding;
    const int shift_src_id = jcp_.typesize_in * (jcp_.dilate_d + 1) * jcp_.ih
            * jcp_.iw * jcp_.ngroups * jcp_.ic_without_padding;
    const int stride_h = signed_input_or_src_zp ? 1 : jcp_.stride_h;
    const int shift_filt_kh
            = jcp_.typesize_in * jcp_.kw * ch_block_all * stride_h;
    const int stride_d = signed_input_or_src_zp ? 1 : jcp_.stride_d;
    const int shift_filt_kd
            = jcp_.typesize_in * jcp_.kw * ch_block_all * jcp_.kh * stride_d;

    Label kd_loop_label, kh_loop_label, skip_kh_loop, skip_kd_loop;
    Label t_overflow_label, no_t_overflow_label, b_overflow_label,
            no_b_overflow_label;
    Label back_overflow_label, no_back_overflow_label, d_h_overflow_label,
            front_overflow_label, no_front_overflow_label, d_h_overflow_label2;
    if (jcp_.ndims == 5) {
        mov(aux_reg_filt_d_, reg_filt_);
        mov(aux_reg_src_d_, reg_src_);

        if (signed_input_or_src_zp) {
            mov(reg_ki_, ptr[param1_ + GET_OFF(back_overflow)]);
            cmp(reg_ki_, 0);
            je(no_back_overflow_label, T_NEAR);

            L(back_overflow_label);
            {
                mov(aux_reg_filt_, aux_reg_filt_d_);
                mov(reg_kh_, jcp_.kh);
                L(d_h_overflow_label);
                {
                    compute_ker(ur_w, 0, 0, last_ic_block_flag, true);
                    add(aux_reg_filt_, shift_filt_kh);
                    dec(reg_kh_);
                    jnz(d_h_overflow_label);
                }

                add(aux_reg_filt_d_, shift_filt_kd);
                dec(reg_ki_);
                jnz(back_overflow_label);
            }
            L(no_back_overflow_label);
        }

        mov(reg_ki_, ptr[param1_ + GET_OFF(kd_padding)]);

        if ((signed_input_or_src_zp) || (jcp_.dilate_d >= jcp_.id)
                || ((!signed_input_or_src_zp)
                        && ((min(jcp_.f_pad, jcp_.back_pad) < 0)
                                || ((jcp_.kd - 1) * (jcp_.dilate_d + 1)
                                        < nstl::max(
                                                jcp_.f_pad, jcp_.back_pad))))) {
            cmp(reg_ki_, 0);
            je(skip_kd_loop, T_NEAR);
        }

        L(kd_loop_label);
        mov(aux_reg_src_, aux_reg_src_d_);
        mov(aux_reg_filt_, aux_reg_filt_d_);
    } else {
        mov(aux_reg_src_, reg_src_);
        mov(aux_reg_filt_, reg_filt_);
    }

    if (signed_input_or_src_zp && jcp_.ndims > 3) {
        /* Weights are transposed, so first compute 'bottom' padding. */
        mov(reg_overflow_, ptr[param1_ + GET_OFF(b_overflow)]);
        cmp(reg_overflow_, 0);
        je(no_b_overflow_label, T_NEAR);
        L(b_overflow_label);
        {
            compute_ker(ur_w, 0, 0, last_ic_block_flag, true);

            add(aux_reg_filt_, shift_filt_kh);
            dec(reg_overflow_);
            cmp(reg_overflow_, 0);
            jg(b_overflow_label, T_NEAR);
        }
        L(no_b_overflow_label);
    }

    mov(reg_kh_, ptr[param1_ + GET_OFF(kh_padding)]);

    if ((signed_input_or_src_zp) || (jcp_.dilate_h >= jcp_.ih)
            || ((!signed_input_or_src_zp)
                    && ((min(jcp_.t_pad, jcp_.b_pad) < 0)
                            || ((jcp_.kh - 1) * (jcp_.dilate_h + 1)
                                    < nstl::max(jcp_.t_pad, jcp_.b_pad))))) {
        cmp(reg_kh_, 0);
        je(skip_kh_loop, T_NEAR);
    }

    L(kh_loop_label);
    {
        compute_ker(ur_w, l_overflow, r_overflow, last_ic_block_flag, false);
        sub(aux_reg_src_, shift_src_ih);
        add(aux_reg_filt_, shift_filt_kh);
        dec(reg_kh_);

        /* Insert weight compensation in stride 'holes' */
        if (signed_input_or_src_zp && jcp_.stride_h > 1) {
            Label kh_comp_loop;

            cmp(reg_kh_, 0);
            je(skip_kh_loop, T_NEAR);
            mov(reg_comp_strides_, jcp_.stride_h - 1);
            L(kh_comp_loop);
            {
                compute_ker(ur_w, 0, 0, last_ic_block_flag, true);
                add(aux_reg_filt_, shift_filt_kh);
                dec(reg_comp_strides_);
                cmp(reg_comp_strides_, 0);
                jg(kh_comp_loop, T_NEAR);
            }
        }
        cmp(reg_kh_, 0);
        jg(kh_loop_label, T_NEAR);
    }
    L(skip_kh_loop);
    if (signed_input_or_src_zp && jcp_.ndims > 3) {
        mov(reg_overflow_, ptr[param1_ + GET_OFF(t_overflow)]);
        cmp(reg_overflow_, 0);
        je(no_t_overflow_label, T_NEAR);
        L(t_overflow_label);
        {
            compute_ker(ur_w, 0, 0, last_ic_block_flag, true);

            add(aux_reg_filt_, shift_filt_kh);
            dec(reg_overflow_);
            cmp(reg_overflow_, 0);
            jg(t_overflow_label, T_NEAR);
        }
        L(no_t_overflow_label);
    }

    if (jcp_.ndims == 5) {
        sub(aux_reg_src_d_, shift_src_id);
        add(aux_reg_filt_d_, shift_filt_kd);
        dec(reg_ki_);

        /* Insert weight compensation in stride 'holes' */
        if (signed_input_or_src_zp && jcp_.stride_d > 1) {
            Label kd_comp_loop, kd_kh_comp_loop;
            cmp(reg_ki_, 0);
            jz(skip_kd_loop, T_NEAR);
            mov(reg_comp_strides_, jcp_.stride_d - 1);
            L(kd_comp_loop);
            mov(aux_reg_filt_, aux_reg_filt_d_);
            mov(reg_kh_, jcp_.kh);
            L(kd_kh_comp_loop);
            {
                compute_ker(ur_w, 0, 0, last_ic_block_flag, true);
                add(aux_reg_filt_, shift_filt_kh);
                dec(reg_kh_);
                jnz(kd_kh_comp_loop, T_NEAR);
            }
            add(aux_reg_filt_d_, shift_filt_kd);
            dec(reg_comp_strides_);
            jnz(kd_comp_loop);
        }

        cmp(reg_ki_, 0);
        jg(kd_loop_label, T_NEAR);
        L(skip_kd_loop);
        if (signed_input_or_src_zp) {
            mov(reg_ki_, ptr[param1_ + GET_OFF(f_overflow)]);
            cmp(reg_ki_, 0);
            jz(no_front_overflow_label, T_NEAR);
            L(front_overflow_label);
            {
                mov(aux_reg_filt_, aux_reg_filt_d_);
                mov(reg_kh_, jcp_.kh);
                L(d_h_overflow_label2);
                {
                    compute_ker(ur_w, 0, 0, last_ic_block_flag, true);
                    add(aux_reg_filt_, shift_filt_kh);
                    dec(reg_kh_);
                    jnz(d_h_overflow_label2);
                }
                add(aux_reg_filt_d_, shift_filt_kd);
                dec(reg_ki_);
                jnz(front_overflow_label);
            }
            L(no_front_overflow_label);
        }
    }
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_deconv_fwd_kernel<isa, Vmm>::prepare_output(int ur_w) {
    for (int ocb = 0; ocb < jcp_.nb_oc_blocking; ocb++) {
        for (int ur = 0; ur < ur_w; ur++) {
            const Vmm vmm = vmm_out(ur, ocb);
            uni_vpxor(vmm, vmm, vmm);
        }
    }
    if (jcp_.signed_input) {
        const auto xmm_shift = Xbyak::Xmm(vmm_shift_.getIdx());
        mov(reg_scratch_, 0x80808080);
        uni_vmovq(xmm_shift, reg_scratch_);
        uni_vpbroadcastd(vmm_shift_, xmm_shift);
    }
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_deconv_fwd_kernel<isa, Vmm>::cvt2ps(data_type_t type_in,
        const Vmm &vmm_in, const Reg64 &reg, int offset, int load_size) {

    load_data(type_in, vmm_in, reg, offset, load_size);
    if (type_in != data_type::f32) uni_vcvtdq2ps(vmm_in, vmm_in);
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_deconv_fwd_kernel<isa, Vmm>::apply_postops(int ur_w,
        bool last_oc_block, const float *p_sum_scale, const int32_t *p_sum_zp) {
    const auto sum_injector = [=]() {
        if (p_sum_scale) { // post_op: sum
            for (int k = 0; k < jcp_.nb_oc_blocking; k++) {
                const bool mask_flag
                        = last_oc_block == 1 && k == jcp_.nb_oc_blocking - 1;
                for (int j = 0; j < ur_w; j++) {
                    const int aux_output_offset = jcp_.typesize_out
                            * (k * jcp_.oc_block
                                    + j * jcp_.oc_without_padding
                                            * jcp_.ngroups);
                    cvt2ps(jcp_.dst_dt, vmm_prev_dst_, reg_dst_,
                            aux_output_offset,
                            mask_flag ? get_tail_size() : get_blocking_size());
                    if (*p_sum_zp != 0) {
                        uni_vbroadcastss(vmm_sum_zp_, ptr[reg_ptr_sum_zp_]);
                        uni_vcvtdq2ps(vmm_sum_zp_, vmm_sum_zp_);
                        uni_vsubps(vmm_prev_dst_, vmm_prev_dst_, vmm_sum_zp_);
                    }
                    const Vmm vmm = vmm_out(j, k);
                    if (*p_sum_scale == 1.f)
                        uni_vaddps(vmm, vmm, vmm_prev_dst_);
                    else {
                        uni_vbroadcastss(vmm_tmp_, ptr[reg_ptr_sum_scale_]);
                        uni_vfmadd231ps(vmm, vmm_prev_dst_, vmm_tmp_);
                    }
                }
            }
        }
    };

    if (p_sum_scale)
        postops_injector_->set_lambda_injector(
                primitive_kind::sum, sum_injector);

    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
    if (jcp_.with_binary) {
        for (int ocb = 0; ocb < jcp_.nb_oc_blocking; ocb++) {
            const bool mask_flag
                    = last_oc_block && ocb == jcp_.nb_oc_blocking - 1;
            for (int ur = 0; ur < ur_w; ur++) {
                const int vmm_idx = vmm_out(ur, ocb).getIdx();
                rhs_arg_params.vmm_idx_to_oc_elem_off_addr.emplace(
                        vmm_idx, ptr[param1_ + GET_OFF(oc_l_off)]);
                rhs_arg_params.vmm_idx_to_oc_elem_off_val.emplace(
                        vmm_idx, ocb * jcp_.oc_block);
                if (mask_flag) rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
            }
        }
    }

    std::map<size_t, int> vmm_idx_off;
    for (int ocb = 0; ocb < jcp_.nb_oc_blocking; ocb++) {
        for (int ur = 0; ur < ur_w; ur++) {
            vmm_idx_off.insert({vmm_out(ur, ocb).getIdx(), ocb * jcp_.oc_block * sizeof(float)});
        }
    }
    depthwise_injector::dynamic_params_t ddp {vmm_d_weights.getIdx(), vmm_d_bias.getIdx(), reg_d_weights, reg_d_bias,
                                              ptr[this->param1 + GET_OFF(oc_off)], vmm_idx_off,
                                              this->rsp, base_post_ops_data_offset};
    quantization_injector::dynamic_params_t qdp {ptr[this->param1 + GET_OFF(oc_off)], vmm_idx_off, jcp_.dst_dt,
                                                 this->rsp, base_post_ops_data_offset};

    const int nb_oc_block
            = jcp_.is_depthwise ? jcp_.nb_ch_blocking : jcp_.nb_oc_blocking;
    postops_injector_->compute_vector_range(
            16 - nb_oc_block * ur_w, 16, rhs_arg_params, ddp, qdp);
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_deconv_fwd_kernel<isa, Vmm>::store_output(
        int ur_w, bool last_oc_block) {
    mov(reg_bias_, ptr[param1_ + GET_OFF(bias)]);
    mov(reg_ptr_scales_, ptr[param1_ + GET_OFF(scales)]);

    if (jcp_.signed_input)
        mov(reg_compensation_, ptr[param1_ + GET_OFF(compensation)]);

    if (jcp_.src_zero_point) {
        mov(reg_zp_src_, ptr[param1_ + GET_OFF(src_zero_point)]);
        mov(reg_zp_compensation_, ptr[param1_ + GET_OFF(zp_compensation)]);
    }

    const auto &p = jcp_.post_ops;
    const int sum_idx = p.find(primitive_kind::sum);
    const float *p_sum_scale
            = (sum_idx != -1) ? &p.entry_[sum_idx].sum.scale : nullptr;
    const int32_t *p_sum_zp
            = (sum_idx != -1) ? &p.entry_[sum_idx].sum.zero_point : nullptr;

    if (jcp_.with_bias && jcp_.signed_input && jcp_.ver != ver_vnni) {
        mov(reg_bias_alpha_, float2int(jcp_.wei_adj_scale));
        uni_vmovq(xmm_bias_alpha(), reg_bias_alpha_);
        uni_vbroadcastss(vmm_bias_alpha(), xmm_bias_alpha());
    }

    if (jcp_.src_zero_point) {
        const auto &vmm_src_zp = vmm_tmp_;
        const auto &vmm_zp_comp = vmm_scale_;
        uni_vbroadcastss(vmm_src_zp, ptr[reg_zp_src_]);

        for (int ocb = 0; ocb < jcp_.nb_oc_blocking; ocb++) {
            const int zp_offset = sizeof(int32_t) * ocb * jcp_.oc_block;
            const bool mask_flag
                    = last_oc_block && ocb == jcp_.nb_oc_blocking - 1;
            const int load_size
                    = mask_flag ? get_tail_size() : get_blocking_size();
            load_data(data_type::s32, vmm_zp_comp, reg_zp_compensation_,
                    zp_offset, load_size);
            uni_vpmulld(vmm_zp_comp, vmm_zp_comp, vmm_src_zp);

            for (int ur = 0; ur < ur_w; ur++) {
                const auto vmm_dst = vmm_out(ur, ocb);
                uni_vpaddd(vmm_dst, vmm_dst, vmm_zp_comp);
            }
        }
    }

    for (int ocb = 0; ocb < jcp_.nb_oc_blocking; ocb++) {
        const bool mask_flag = last_oc_block && ocb == jcp_.nb_oc_blocking - 1;
        const int scale_offset
                = jcp_.is_oc_scale * (sizeof(float) * ocb * jcp_.oc_block);

        const auto vmm_bias_ = vmm_tmp_;
        if (jcp_.with_bias) {
            int bias_offset = jcp_.typesize_bia * ocb * jcp_.oc_block;
            cvt2ps(jcp_.bia_dt, vmm_bias_, reg_bias_, bias_offset,
                    mask_flag ? get_tail_size() : get_blocking_size());
            if (jcp_.signed_input && jcp_.ver != ver_vnni)
                uni_vmulps(vmm_bias_, vmm_bias_, vmm_bias_alpha());
        }
        if (jcp_.signed_input) {
            const int comp_offset = sizeof(int32_t) * ocb * jcp_.oc_block;
            cvt2ps(data_type::s32, vmm_comp_, reg_compensation_, comp_offset,
                    mask_flag ? get_tail_size() : get_blocking_size());
        }

        /* add to ymm_accum: compensation, bias */
        uni_vmovups(vmm_scale_, ptr[reg_ptr_scales_ + scale_offset]);
        for (int ur = 0; ur < ur_w; ur++) {
            const Vmm vmm = vmm_out(ur, ocb);
            uni_vcvtdq2ps(vmm, vmm);
            if (jcp_.signed_input) uni_vaddps(vmm, vmm, vmm_comp_);
            if (jcp_.with_bias) uni_vaddps(vmm, vmm, vmm_bias_);
            uni_vmulps(vmm, vmm, vmm_scale_);
        }
    }

    if (p_sum_scale && *p_sum_scale != 1.f)
        mov(reg_ptr_sum_scale_, reinterpret_cast<size_t>(p_sum_scale));
    if (p_sum_zp && *p_sum_zp != 0) {
        mov(reg_ptr_sum_zp_, reinterpret_cast<size_t>(p_sum_zp));
    }
    if (jcp_.with_eltwise || jcp_.with_binary || jcp_.with_sum || jcp_.with_depthwise || jcp_.with_quantization)
        apply_postops(ur_w, last_oc_block, p_sum_scale, p_sum_zp);
    if (jcp_.dst_zero_point) {
        mov(reg_zp_dst_, ptr[param1_ + GET_OFF(dst_zero_point)]);
        const auto &vmm_zp_dst = vmm_tmp_;
        uni_vbroadcastss(vmm_zp_dst, ptr[reg_zp_dst_]);
        uni_vcvtdq2ps(vmm_zp_dst, vmm_zp_dst);

        for_(int ocb = 0; ocb < jcp_.nb_oc_blocking; ocb++)
        for (int ur = 0; ur < ur_w; ur++) {
            const auto vmm_dst = vmm_out(ur, ocb);
            uni_vaddps(vmm_dst, vmm_dst, vmm_zp_dst);
        }
    }

    // Properly saturate the accumulators for integer datatypes

    // No need to saturate on lower bound for signed integer types, as
    // the conversion to int would return INT_MIN, and then proper
    // saturation will happen when storing data
    if (jcp_.dst_dt == data_type::u8) {
        uni_vpxor(vmm_zero_, vmm_zero_, vmm_zero_);
        for (int ocb = 0; ocb < jcp_.nb_oc_blocking; ocb++) {
            for (int ur = 0; ur < ur_w; ur++) {
                const Vmm vmm = vmm_out(ur, ocb);
                uni_vmaxps(vmm, vmm, vmm_zero_);
            }
        }
    }

    if (one_of(jcp_.dst_dt, data_type::u8, data_type::s8, data_type::s32)) {
        float saturation_ubound = types::max_value<float>(jcp_.dst_dt);
        const Xmm xmm_saturation(vmm_saturation_.getIdx());
        mov(reg_ptr_saturation_ubound_, float2int(saturation_ubound));
        uni_vmovq(xmm_saturation, reg_ptr_saturation_ubound_);
        uni_vbroadcastss(vmm_saturation_, xmm_saturation);

        for (int ocb = 0; ocb < jcp_.nb_oc_blocking; ocb++) {
            for (int ur = 0; ur < ur_w; ur++) {
                const Vmm vmm = vmm_out(ur, ocb);
                uni_vminps(vmm, vmm, vmm_saturation_);
            }
        }
    }

    if (one_of(jcp_.dst_dt, data_type::u8, data_type::s8, data_type::s32)) {
        for (int ocb = 0; ocb < jcp_.nb_oc_blocking; ocb++) {
            for (int ur = 0; ur < ur_w; ur++) {
                const Vmm vmm = vmm_out(ur, ocb);
                uni_vcvtps2dq(vmm, vmm);
            }
        }
    }

    /* write out register to output_addr */
    for (int ocb = 0; ocb < jcp_.nb_oc_blocking; ocb++) {
        const bool mask_flag = last_oc_block && ocb == jcp_.nb_oc_blocking - 1;
        for (int ur = 0; ur < ur_w; ur++) {
            const int aux_dst_off = jcp_.typesize_out
                    * (ur * jcp_.ngroups * jcp_.oc_without_padding
                            + ocb * jcp_.oc_block);
            const Vmm r_vmm = vmm_out(ur, ocb);
            store_data(jcp_.dst_dt, r_vmm, reg_dst_, aux_dst_off,
                    mask_flag ? get_tail_size() : get_blocking_size());
        }
    }
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_deconv_fwd_kernel<isa, Vmm>::icb_loop(
        int ur_w, int l_overflow, int r_overflow, bool is_last_sp_block) {

    const int shift_src_icb = jcp_.typesize_in * jcp_.ic_block;
    const size_t shift_filt_icb = (size_t)jcp_.typesize_in * jcp_.kd * jcp_.kh
            * jcp_.kw * jcp_.ic_block * jcp_.oc_block;

    prepare_output(ur_w);

    Label skip_icb_loop, icb_loop_label;

    mov(reg_icb_, jcp_.nb_ic);
    mov(reg_oc_blocks_, ptr[param1_ + GET_OFF(oc_blocks)]);

    if (zp::should_calculate_deconv_zp_src_pad_str_comp(jcp_)
            && jcp_.ndims > 3) {
        mov(reg_scratch_,
                qword[param1_ + GET_OFF(zp_src_pad_str_compensation)]);
        mov(zp_src_pad_comp_addr_, reg_scratch_);
    }

    L(icb_loop_label);
    {
        if (jcp_.ngroups % jcp_.ch_block != 0
                || jcp_.ic_without_padding != jcp_.ic) {
            Label common_ker, end_ker;
            if (jcp_.is_depthwise) {
                cmp(reg_oc_blocks_, jcp_.nb_ch - 1);
                jne(common_ker, T_NEAR);
            } else {
                cmp(reg_icb_, 1);
                jg(common_ker, T_NEAR);
            }

            kh_loop(ur_w, l_overflow, r_overflow, last_sp_block);
            jmp(end_ker, T_NEAR);

            L(common_ker);
            kh_loop(ur_w, l_overflow, r_overflow, no_last_block);

            L(end_ker);
        } else {
            kh_loop(ur_w, l_overflow, r_overflow, no_last_block);
        }

        add(reg_src_, shift_src_icb);
        safe_add(reg_filt_, shift_filt_icb, reg_ker_long_offt_);
        dec(reg_icb_);
        cmp(reg_icb_, 0);
        jg(icb_loop_label, T_NEAR);
    }

    /* come-back pointers */
    sub(reg_src_, jcp_.nb_ic * shift_src_icb);
    safe_sub(reg_filt_, jcp_.nb_ic * shift_filt_icb, reg_ker_long_offt_);
    L(skip_icb_loop);

    if (jcp_.ngroups % jcp_.ch_block != 0
            || jcp_.oc_without_padding != jcp_.oc) {
        Label common_store, end_store;
        if (jcp_.is_depthwise)
            cmp(reg_oc_blocks_, jcp_.nb_ch - 1);
        else
            cmp(reg_oc_blocks_, jcp_.nb_oc - jcp_.nb_oc_blocking);
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

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_deconv_fwd_kernel<isa, Vmm>::generate() {
    preamble();

    if (postops_injector_)
        postops_injector_->push_post_ops_data_on_stack(param1, GET_OFF(post_ops_binary_rhs_arg_vec), reg_src_, reg_filt_);

    if (zp::should_calculate_deconv_zp_src_pad_str_comp(jcp_)) {
        sub(rsp, reserved_stack_size_);
        base_post_ops_data_offset += reserved_stack_size_;
    }

    const auto vmm_one_128 = Xbyak::Xmm(vmm_one_.getIdx());
    mov(reg_scratch_, 0x10001);
    uni_vmovq(vmm_one_128, reg_scratch_);
    uni_vpbroadcastd(vmm_one_, vmm_one_128);

    mov(reg_src_, ptr[param1_ + GET_OFF(src)]);
    mov(reg_filt_, ptr[param1_ + GET_OFF(filt)]);
    mov(reg_dst_, ptr[param1_ + GET_OFF(dst)]);

    const int dst_shift = jcp_.typesize_out * jcp_.ur_w * jcp_.ngroups
            * jcp_.oc_without_padding;
    const int src_shift = jcp_.typesize_in * (jcp_.ur_w / jcp_.stride_w)
            * jcp_.ngroups * jcp_.ic_without_padding;

    const int l_overflow = max(0,
            ((jcp_.kw - 1) * (jcp_.dilate_w + 1) - jcp_.l_pad) / jcp_.stride_w);
    const int r_overflow = max(0,
            ((jcp_.kw - 1) * (jcp_.dilate_w + 1) - max(0, jcp_.r_pad))
                    / jcp_.stride_w);

    const int r_overflow1 = nstl::max(0,
            ((jcp_.kw - 1) * (jcp_.dilate_w + 1) - nstl::max(0, jcp_.r_pad)
                    - jcp_.ur_w_tail)
                    / jcp_.stride_w);
    int nur_w = jcp_.ow / jcp_.ur_w;
    if (r_overflow1 > 0) nur_w--;

    if (jcp_.ur_w == jcp_.ow) {
        icb_loop(jcp_.ur_w, l_overflow, r_overflow, true);
    } else if (nur_w == 0) {
        icb_loop(jcp_.ur_w, l_overflow, r_overflow1, jcp_.ur_w_tail == 0);
        add(reg_src_, src_shift);
        add(reg_dst_, dst_shift);
        if (jcp_.ur_w_tail != 0) icb_loop(jcp_.ur_w_tail, 0, r_overflow, true);
    } else {
        xor_(reg_nur_w_, reg_nur_w_);
        if (l_overflow > 0) {
            icb_loop(jcp_.ur_w, l_overflow, 0, false);
            add(reg_src_, src_shift);
            add(reg_dst_, dst_shift);
            inc(reg_nur_w_);
        }
        if ((l_overflow <= 0 && nur_w > 0) || (l_overflow > 0 && nur_w > 1)) {
            Label ow_loop_label;
            L(ow_loop_label);
            {
                icb_loop(jcp_.ur_w, 0, 0, false);
                add(reg_src_, src_shift);
                add(reg_dst_, dst_shift);
                inc(reg_nur_w_);
                cmp(reg_nur_w_, nur_w);
                jl(ow_loop_label, T_NEAR);
            }
        }
        if (r_overflow1 > 0) {
            icb_loop(jcp_.ur_w, 0, r_overflow1, jcp_.ur_w_tail == 0);
            add(reg_src_, src_shift);
            add(reg_dst_, dst_shift);
        }
        if (jcp_.ur_w_tail != 0) {
            icb_loop(jcp_.ur_w_tail, 0, r_overflow, true);
        }
    }

    if (zp::should_calculate_deconv_zp_src_pad_str_comp(jcp_)) {
        add(rsp, reserved_stack_size_);
        base_post_ops_data_offset -= reserved_stack_size_;
    }

    if (postops_injector_)
        postops_injector_->reset_stack_pointer();

    postamble();

    if (jcp_.with_eltwise) postops_injector_->prepare_table();
}

template <cpu_isa_t isa>
jit_uni_x8s8s32x_deconvolution_fwd_t<isa>::jit_uni_x8s8s32x_deconvolution_fwd_t(
        const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
jit_uni_x8s8s32x_deconvolution_fwd_t<
        isa>::~jit_uni_x8s8s32x_deconvolution_fwd_t()
        = default;

template <cpu_isa_t isa>
status_t jit_uni_x8s8s32x_deconvolution_fwd_t<isa>::pd_t::init(
        engine_t *engine) {
    using namespace data_type;
    using skip_mask_t = primitive_attr_t::skip_mask_t;
    const bool ok = true && is_fwd()
            && (desc()->alg_kind & alg_kind::deconvolution_direct)
            && utils::one_of(src_md(0)->data_type, s8, u8)
            && weights_md(0)->data_type == s8
            && IMPLICATION(with_bias(),
                    utils::one_of(weights_md(1)->data_type, f32, s32, s8, u8))
            && utils::one_of(dst_md(0)->data_type, f32, s32, s8, u8)
            && desc()->accum_data_type == s32
            && attr()->has_default_values(skip_mask_t::oscale
                    | skip_mask_t::post_ops | skip_mask_t::zero_points_runtime);
    if (!ok) return status::unimplemented;

    CHECK(jit_uni_x8s8s32x_deconv_fwd_kernel<isa>::init_conf(jcp_, *desc(),
            src_md_, weights_md_, dst_md_, with_bias(), bias_md_, attr_,
            dnnl_get_max_threads()));

    auto scratchpad = scratchpad_registry().registrar();
    jit_uni_x8s8s32x_deconv_fwd_kernel<isa>::init_scratchpad(
            scratchpad, jcp_, *attr());

    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_x8s8s32x_deconvolution_fwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_,
            new jit_uni_x8s8s32x_deconv_fwd_kernel<isa>(pd()->jcp_,
                    *pd()->attr(), memory_desc_wrapper(pd()->dst_md()))));

    if (zp::should_calculate_deconv_zp_src_pad_str_comp(pd()->jcp_)) {
        CHECK(safe_ptr_assign(zp_src_pad_comp_kernel_,
                zp::create_deconv_zp_pad_str_comp_ker<isa>(pd()->jcp_)));
        const auto zp_kernel_status = zp_src_pad_comp_kernel_->create_kernel();
        if (zp_kernel_status != status::success) return zp_kernel_status;
    }

    return kernel_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_x8s8s32x_deconvolution_fwd_t<isa>::execute(
        const exec_ctx_t &ctx) const {
    const auto &_pd = pd();
    const auto &ndims = _pd->ndims();
    if (ndims == 3)
        return execute_forward_1d(ctx);
    else if (ndims == 4)
        return execute_forward_2d(ctx);
    else if (ndims == 5)
        return execute_forward_3d(ctx);
    else
        return status::unimplemented;
    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_x8s8s32x_deconvolution_fwd_t<isa>::execute_forward_1d(
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

    const size_t dst_dt_size = types::data_type_size(dst_d.data_type());

    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(jcp.post_ops, ctx);
    auto scratchpad = ctx.get_scratchpad_grantor();
    int32_t *zp_src_comp_scratch = scratchpad.get<int32_t>(key_deconv_zp);

    if (zp::should_calculate_deconv_zp_src_pad_str_comp(jcp))
        zp::compute_deconv_zp_pad_str_comp_ker(jcp, pd()->with_groups(),
                weights_d, weights, zp_src, zp_src_comp_scratch,
                zp_src_pad_comp_kernel_.get());

    const int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    const int nb_groups = jcp.nb_ch;

    const float *oscales = pd()->attr()->output_scales_.scales_;
    if (jcp.signed_input && jcp.ver != ver_vnni) {
        auto local_scales = ctx.get_scratchpad_grantor().template get<float>(
                key_conv_adjusted_scales);
        const size_t count = pd()->attr()->output_scales_.count_;
        const float factor = 1.f / pd()->jcp_.wei_adj_scale;
        if (count == 1) {
            utils::array_set(local_scales, oscales[0] * factor, 8);
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
        const int work_amount = jcp.mb * nb_groups * oc_chunks;
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

            const int ocb = occ * jcp.nb_oc_blocking;
            const int g_oc
                    = (g * jcp.ch_block * jcp.nb_oc + ocb) * jcp.oc_block;
            const int g_ic = g * jcp.ch_block * jcp.ic;

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
            p.zp_compensation
                    = jcp.src_zero_point ? zp_compensation + g_oc : nullptr;
            p.zp_src_pad_str_compensation = zp_src_comp_scratch
                    ? zp_src_comp_scratch + g_oc
                    : nullptr;
            p.src_zero_point = zp_src;
            p.dst_zero_point = zp_dst;

            p.oc_off = g_oc * sizeof(float);

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

template <cpu_isa_t isa>
status_t jit_uni_x8s8s32x_deconvolution_fwd_t<isa>::execute_forward_2d(
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
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(jcp.post_ops, ctx);

    auto scratchpad = ctx.get_scratchpad_grantor();
    int32_t *zp_src_comp_scratch = scratchpad.get<int32_t>(key_deconv_zp);

    if (zp::should_calculate_deconv_zp_src_pad_str_comp(jcp))
        zp::compute_deconv_zp_pad_str_comp_ker(jcp, pd()->with_groups(),
                weights_d, weights, zp_src, zp_src_comp_scratch,
                zp_src_pad_comp_kernel_.get());

    const int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    const int nb_groups = jcp.nb_ch;

    const size_t src_h_stride = src_d.blk_off(0, 0, 1);
    const size_t dst_h_stride = dst_d.blk_off(0, 0, 1);
    const size_t wht_kh_stride = wht_blk_off(weights_d, 0, 0, 0, 1);

    const float *oscales = pd()->attr()->output_scales_.scales_;
    if (jcp.signed_input && jcp.ver != ver_vnni) {
        auto local_scales = ctx.get_scratchpad_grantor().template get<float>(
                key_conv_adjusted_scales);
        const size_t count = pd()->attr()->output_scales_.count_;
        const float factor = 1.f / pd()->jcp_.wei_adj_scale;
        if (count == 1) {
            utils::array_set(local_scales, oscales[0] * factor, 8);
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
        const int work_amount = jcp.mb * nb_groups * oc_chunks * jcp.oh;
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

            const int ocb = occ * jcp.nb_oc_blocking;
            const int g_oc
                    = (g * jcp.ch_block * jcp.nb_oc + ocb) * jcp.oc_block;
            const int g_ic = g * jcp.ch_block * jcp.ic;
            const int work_rem = end - start;
            const int oh_e
                    = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;

            const auto dst_w = dst + dst_dt_size * dst_d.blk_off(n, g_oc);
            const auto src_w = src + src_d.blk_off(n, g_ic);
            const auto wht_w = weights + wht_blk_off(weights_d, g, ocb, 0);
            const auto bias_w = jcp.with_bias
                    ? bias + (bias_d.blk_off(g_oc) * jcp.typesize_bia)
                    : nullptr;
            const int32_t *compensation_w
                    = (jcp.signed_input) ? compensation + g_oc : nullptr;

            const auto scales = &oscales[jcp.is_oc_scale * g_oc];
            for (int oj = oh_s; oj < oh_e; oj++) {
                int ih_max = 0, kh_lo = 0, kh_len = 0;
                if (jcp.dilate_h != 0 && jcp.stride_h == 1) {
                    /* dilation */
                    const int dilate_h = jcp.dilate_h + 1;
                    // Note: use div_up to account for "holes" in filter
                    const int o_t_overflow = div_up(
                            max(0, (jcp.kh - 1) * dilate_h - oj - jcp.t_pad),
                            dilate_h);
                    const int o_b_overflow
                            = div_up(max(0,
                                             (jcp.kh - 1) * dilate_h + 1
                                                     - jcp.oh + oj - jcp.b_pad),
                                    dilate_h);
                    kh_len = jcp.kh - o_t_overflow - o_b_overflow;
                    kh_lo = o_b_overflow;
                    ih_max = oj + jcp.t_pad - o_b_overflow * dilate_h;
                } else {
                    const int o_t_overflow = max(
                            0, (jcp.kh - (oj + 1 + jcp.t_pad)) / jcp.stride_h);
                    const int o_b_overflow = max(0,
                            ((oj + jcp.kh) - (jcp.oh + jcp.b_pad))
                                    / jcp.stride_h);
                    const int overflow_kh_hi = jcp.kh - 1
                            - modulo(jcp.oh + jcp.b_pad - (oj + 1),
                                    jcp.stride_h);
                    const int overflow_kh_lo = (oj + jcp.t_pad) % jcp.stride_h;

                    kh_len = (overflow_kh_hi - overflow_kh_lo) / jcp.stride_h
                            + 1 - o_t_overflow - o_b_overflow;
                    kh_lo = overflow_kh_lo + o_b_overflow * jcp.stride_h;
                    ih_max = (oj + jcp.t_pad - kh_lo) / jcp.stride_h;
                }

                const int wei_stride
                        = (!jcp.signed_input && !jcp.src_zero_point)
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
                p.zp_compensation
                        = jcp.src_zero_point ? zp_compensation + g_oc : nullptr;
                p.zp_src_pad_str_compensation = jcp.src_zero_point
                        ? zp_src_comp_scratch + g_oc
                        : nullptr;
                p.src_zero_point = zp_src;
                p.dst_zero_point = zp_dst;

                p.oc_off = g_oc * sizeof(float);

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

template <cpu_isa_t isa>
status_t jit_uni_x8s8s32x_deconvolution_fwd_t<isa>::execute_forward_3d(
        const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    const auto weights = CTX_IN_MEM(const int8_t *, DNNL_ARG_WEIGHTS);
    const auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    const auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);
    DEFINE_ZERO_POINTS_BUFFER(zp_src, DNNL_ARG_SRC);
    DEFINE_ZERO_POINTS_BUFFER(zp_dst, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const size_t dst_dt_size = types::data_type_size(dst_d.data_type());

    const auto &jcp = pd()->jcp_;
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(jcp.post_ops, ctx);

    const auto scratchpad = ctx.get_scratchpad_grantor();
    int32_t *const zp_src_comp_scratch = scratchpad.get<int32_t>(key_deconv_zp);

    if (zp::should_calculate_deconv_zp_src_pad_str_comp(jcp))
        zp::compute_deconv_zp_pad_str_comp_ker(jcp, pd()->with_groups(),
                weights_d, weights, zp_src, zp_src_comp_scratch,
                zp_src_pad_comp_kernel_.get());

    const int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    const int &nb_groups = jcp.nb_ch;

    const size_t src_d_stride = src_d.blk_off(0, 0, 1);
    const size_t src_h_stride = src_d.blk_off(0, 0, 0, 1);
    const size_t dst_d_stride = dst_d.blk_off(0, 0, 1);
    const size_t dst_h_stride = dst_d.blk_off(0, 0, 0, 1);
    const size_t wht_kd_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
    const size_t wht_kh_stride = wht_blk_off(weights_d, 0, 0, 0, 0, 1);

    const float *oscales = pd()->attr()->output_scales_.scales_;
    if (jcp.signed_input && jcp.ver != ver_vnni) {
        const auto local_scales
                = ctx.get_scratchpad_grantor().template get<float>(
                        key_conv_adjusted_scales);
        const size_t count = pd()->attr()->output_scales_.count_;
        const float factor = 1.f / pd()->jcp_.wei_adj_scale;
        if (count == 1) {
            utils::array_set(local_scales, oscales[0] * factor, 8);
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

            const int ocb = occ * jcp.nb_oc_blocking;
            const int g_oc
                    = (g * jcp.ch_block * jcp.nb_oc + ocb) * jcp.oc_block;
            const int g_ic = g * jcp.ch_block * jcp.ic;
            const int work_rem = end - start;
            const int oh_e
                    = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
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
                const int d_t_overflow = max(
                        0, (jcp.kd - (od_s + 1 + jcp.f_pad)) / jcp.stride_d);
                const int d_back_overflow = max(0,
                        ((od_s + jcp.kd) - (jcp.od + jcp.back_pad))
                                / jcp.stride_d);
                const int overflow_kd_hi = jcp.kd - 1
                        - modulo(jcp.od + jcp.back_pad - (od_s + 1),
                                jcp.stride_d);
                const int overflow_kd_lo = (od_s + jcp.f_pad) % jcp.stride_d;

                kd_len = (overflow_kd_hi - overflow_kd_lo) / jcp.stride_d + 1
                        - d_t_overflow - d_back_overflow;
                kd_lo = overflow_kd_lo + d_back_overflow * jcp.stride_d;
                input_d_s = (od_s + jcp.f_pad - kd_lo) / jcp.stride_d;
            }

            auto dst_w = dst
                    + dst_dt_size
                            * (dst_d.blk_off(n, g_oc) + od_s * dst_d_stride);
            const auto src_w
                    = src + src_d.blk_off(n, g_ic) + input_d_s * src_d_stride;
            const auto wht_w = weights + wht_blk_off(weights_d, g, ocb, 0)
                    + ((jcp.signed_input || jcp.src_zero_point) ? 0 : kd_lo)
                            * wht_kd_stride;
            const auto bias_w = jcp.with_bias
                    ? bias + (bias_d.blk_off(g_oc) * jcp.typesize_bia)
                    : nullptr;
            const int32_t *compensation_w
                    = (jcp.signed_input) ? compensation + g_oc : nullptr;

            const auto scales = &oscales[jcp.is_oc_scale * g_oc];

            for (int oj = oh_s; oj < oh_e; oj++) {
                int ih_max = 0, kh_lo = 0, kh_len = 0;
                if (jcp.dilate_h != 0 && jcp.stride_h == 1) {
                    /* dilation */
                    const int dilate_h = jcp.dilate_h + 1;
                    // Note: use div_up to account for "holes" in filter
                    const int o_t_overflow = div_up(
                            max(0, (jcp.kh - 1) * dilate_h - oj - jcp.t_pad),
                            dilate_h);
                    const int o_b_overflow
                            = div_up(max(0,
                                             (jcp.kh - 1) * dilate_h + 1
                                                     - jcp.oh + oj - jcp.b_pad),
                                    dilate_h);
                    kh_len = jcp.kh - o_t_overflow - o_b_overflow;
                    kh_lo = o_b_overflow;
                    ih_max = oj + jcp.t_pad - o_b_overflow * dilate_h;
                } else {
                    const int o_t_overflow = max(
                            0, (jcp.kh - (oj + 1 + jcp.t_pad)) / jcp.stride_h);
                    const int o_b_overflow = max(0,
                            ((oj + jcp.kh) - (jcp.oh + jcp.b_pad))
                                    / jcp.stride_h);
                    const int overflow_kh_hi = jcp.kh - 1
                            - modulo(jcp.oh + jcp.b_pad - (oj + 1),
                                    jcp.stride_h);
                    const int overflow_kh_lo = (oj + jcp.t_pad) % jcp.stride_h;

                    kh_len = (overflow_kh_hi - overflow_kh_lo) / jcp.stride_h
                            + 1 - o_t_overflow - o_b_overflow;
                    kh_lo = overflow_kh_lo + o_b_overflow * jcp.stride_h;
                    ih_max = (oj + jcp.t_pad - kh_lo) / jcp.stride_h;
                }

                const int wei_stride
                        = (!jcp.signed_input && !jcp.src_zero_point)
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
                p.zp_compensation
                        = jcp.src_zero_point ? zp_compensation + g_oc : nullptr;
                p.zp_src_pad_str_compensation = jcp.src_zero_point
                        ? zp_src_comp_scratch + g_oc
                        : nullptr;
                p.src_zero_point = zp_src;
                p.dst_zero_point = zp_dst;

                p.oc_off = g_oc * sizeof(float);

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

using namespace data_type;
template struct jit_uni_x8s8s32x_deconvolution_fwd_t<avx2>;
template struct jit_uni_x8s8s32x_deconvolution_fwd_t<sse41>;
template struct jit_uni_x8s8s32x_deconv_fwd_kernel<avx2>;
template struct jit_uni_x8s8s32x_deconv_fwd_kernel<sse41>;
template struct _jit_uni_x8s8s32x_deconv_fwd_kernel<avx2, Xbyak::Ymm>;
template struct _jit_uni_x8s8s32x_deconv_fwd_kernel<avx2, Xbyak::Xmm>;
template struct _jit_uni_x8s8s32x_deconv_fwd_kernel<sse41, Xbyak::Xmm>;
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
