/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "cpu_memory.hpp"

#include "jit_avx512_core_u8s8s32x_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

namespace {
void pick_loop_order(jit_conv_conf_t &jcp)
{
    jcp.loop_order = loop_cgn;
    if (jcp.ngroups > 1)
        jcp.loop_order = loop_ngc;
}
}

bool jit_avx512_core_u8s8s32x_fwd_kernel::maybe_relu(int position)
{
    using namespace primitive_kind;
    const auto &p = attr_.post_ops_;

    if (position == 0) {
        /* relu before sum */
        return false
            || jcp.with_eltwise
            || p.contain(eltwise, 0)
            || (jcp.dst_dt == data_type::u8 && !p.contain(sum, 0));
    } else if (position == 1) {
        /* relu after sum */
        const int sum_idx = p.contain(sum, 0)
            ? 0 : (p.contain(sum, 1) ? 1 : -1);
        if (sum_idx == -1)
            return false;

        return false
            || p.contain(eltwise, sum_idx + 1)
            || jcp.dst_dt == data_type::u8;
    }

    return false;
}

void jit_avx512_core_u8s8s32x_fwd_kernel::prepare_output(int ur_w)
{
    Label l_first_load, l_ret;

    mov(reg_channel, ptr[param1 + GET_OFF(channel)]);
    cmp(reg_channel, 0); // FISRT load
    je(l_first_load, T_NEAR);

    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            int offset = jcp.typesize_acc * (k*ur_w + j) * jcp.oc_block;
            vmovups(zmm, EVEX_compress_addr(reg_acc_s32, offset));
        }
    jmp(l_ret, T_NEAR);

    L(l_first_load);
    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            vpxord(zmm, zmm, zmm);
        }

    L(l_ret);
}

void jit_avx512_core_u8s8s32x_fwd_kernel::store_output(int ur_w)
{
    Label l_update_acc, l_ret;

    mov(reg_channel, ptr[param1 + GET_OFF(channel)]);

    int adjusment = jcp.nb_ic - ((jcp.nb_ic_blocking <= 1)
        ? 0
        : jcp.nb_ic_blocking) - 1;
    cmp(reg_channel, adjusment); // LAST channel
    jl(l_update_acc, T_NEAR);

    mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
    mov(reg_ptr_scales, ptr[param1 + GET_OFF(scales)]);

    const auto &p = attr_.post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    const float *p_sum_scale = (sum_idx != -1)
            ? &p.entry_[sum_idx].sum.scale
            : nullptr;
    if (p_sum_scale && *p_sum_scale != 1.f)
        mov(reg_ptr_sum_scale, (size_t)p_sum_scale);

    vpxord(zmm_zero, zmm_zero, zmm_zero);
    for (int k = 0; k < jcp.nb_oc_blocking; k++) {
        int scale_offset = jcp.is_oc_scale * (sizeof(float) * k * jcp.oc_block);
        auto zmm_bias = zmm_tmp;
        if (jcp.with_bias) {
            int bias_offset = jcp.typesize_bia * k * jcp.oc_block;
            auto bias_addr = EVEX_compress_addr(reg_bias, bias_offset);
            switch (jcp.bia_dt) {
            case data_type::f32:
            case data_type::s32: vmovups(zmm_bias, bias_addr); break;
            case data_type::s8: vpmovsxbd(zmm_bias, bias_addr); break;
            case data_type::u8: vpmovzxbd(zmm_bias, bias_addr); break;
            default: assert(!"unsupported dst data type");
            }
            if (jcp.bia_dt != data_type::f32)
                vcvtdq2ps(zmm_bias, zmm_bias);
        }
        for (int j = 0; j < ur_w; j++) {
            int aux_output_offset
                = jcp.typesize_out * (k * jcp.oc_block
                                        + j * jcp.oc * jcp.ngroups);
            auto addr = EVEX_compress_addr(reg_out, aux_output_offset);

            Xmm xmm = xmm_out(j, k);
            Zmm zmm = zmm_out(j, k);
            vcvtdq2ps(zmm, zmm);
            if (jcp.with_bias)
                vaddps(zmm, zmm, zmm_bias);
            vmulps(zmm, zmm, EVEX_compress_addr(reg_ptr_scales, scale_offset));
            if (maybe_relu(0))
                vmaxps(zmm, zmm_zero, zmm);
            if (p_sum_scale) { // post_op: sum
                auto zmm_prev_dst = zmm_bcast;
                switch (jcp.dst_dt) {
                case data_type::f32:
                case data_type::s32: vmovups(zmm_prev_dst, addr); break;
                case data_type::s8: vpmovsxbd(zmm_prev_dst, addr); break;
                case data_type::u8: vpmovzxbd(zmm_prev_dst, addr); break;
                default: assert(!"unknown dst_dt");
                }
                if (jcp.dst_dt != data_type::f32)
                    vcvtdq2ps(zmm_prev_dst, zmm_prev_dst);
                if (*p_sum_scale == 1.f)
                    vaddps(zmm, zmm_prev_dst);
                else
                    vfmadd231ps(zmm, zmm_prev_dst, zword_b[reg_ptr_sum_scale]);
            }
            if (maybe_relu(1))
                vmaxps(zmm, zmm_zero, zmm);

            if (jcp.dst_dt != data_type::f32) {
                if (attr_.round_mode_ == round_mode::nearest)
                    vcvtps2dq(zmm | T_rn_sae, zmm);
                else if (attr_.round_mode_ == round_mode::down)
                    vcvtps2dq(zmm | T_rd_sae, zmm);
                else
                    assert(!"unimplemented");
            }
            switch (jcp.dst_dt) {
            case data_type::f32:
            case data_type::s32: vmovups(addr, zmm); break;
            case data_type::s8: vpmovsdb(xmm, zmm); vmovups(addr, xmm); break;
            case data_type::u8: vpmovusdb(xmm, zmm); vmovups(addr, xmm); break;
            default: assert(!"unknown dst_dt");
            }
        }
    }
    jmp(l_ret, T_NEAR);

    L(l_update_acc);
    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            int offset = jcp.typesize_acc * (k*ur_w + j) * jcp.oc_block;
            vmovups(EVEX_compress_addr(reg_acc_s32, offset), zmm);
        }
    L(l_ret);
}

void jit_avx512_core_u8s8s32x_fwd_kernel::compute_loop(int ur_w,
    int pad_l, int pad_r)
{
    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int ch_block_all = jcp.ch_block * ic_block * oc_block;

    int nb_oc_block = jcp.nb_oc_blocking;
    int nb_ic_block = jcp.nb_ic_blocking;

    Label kh_label, skip_kh_loop;

    int shift_kernel_ptr = jcp.typesize_in * jcp.kw * ch_block_all;
    int shift_input_ptr = jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw * jcp.ic
            * jcp.ngroups;

    auto input_offset = [=](int oi, int nb_ic, int ic, int ki) {
        return jcp.typesize_in
                * ((ki * (jcp.dilate_w + 1) + oi * stride_w - pad_l) * jcp.ic
                                  * jcp.ngroups
                          + 4 * ic + nb_ic * jcp.ic_block);
    };
    auto kernel_offset = [=](int ii, int nb_ic, int ic, int ki) {
        return jcp.typesize_in
                * ((ii * jcp.nb_ic * jcp.kh * jcp.kw + ki) * ch_block_all
                          + 4 * ic * oc_block
                          + nb_ic * jcp.kh * jcp.kw * ch_block_all);
    };
    auto compute = [=](Zmm vreg_acc, Zmm vreg_wei, Zmm vreg_src) {
        if (jcp.is_depthwise) {
            vpmulld(zmm_tmp, vreg_src, vreg_wei);
            vpaddd(vreg_acc, vreg_acc, zmm_tmp);
        } else {
            if (jcp.ver == ver_vnni) {
                vpdpbusd(vreg_acc, vreg_src, vreg_wei);
            } else {
                vpmaddubsw(zmm_tmp, vreg_src, vreg_wei);
                vpmaddwd(zmm_tmp, zmm_tmp, zmm_one);
                vpaddd(vreg_acc, vreg_acc, zmm_tmp);
            }
        }
    };

    prepare_output(ur_w);

    mov(aux_reg_inp, reg_inp);
    mov(aux_reg_ker, reg_ker);

    mov(reg_kj, reg_kh);
    if ((jcp.kh - 1) * (jcp.dilate_h + 1) < jcp.t_pad) {
        cmp(reg_kj, 0);
        je(skip_kh_loop, T_NEAR);
    }
    L(kh_label); {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = get_ow_start(ki, pad_l);
            int jj_end = get_ow_end(ur_w, ki, pad_r);

            for (int cc = 0; cc < nb_ic_block; cc++) {
                for (int ic = 0; ic < (jcp.is_depthwise ? 1 : ic_block / 4);
                        ic++) {
                    for (int jj = jj_start; jj < jj_end; jj++) {
                        int aux_input_offset = input_offset(jj, cc, ic, ki);
                        if (jcp.is_depthwise)
                            vpmovzxbd(zmm_inp(jj, nb_oc_block),
                                    EVEX_compress_addr(
                                              aux_reg_inp, aux_input_offset));
                        else
                            vpbroadcastd(zmm_inp(jj, nb_oc_block),
                                    EVEX_compress_addr(aux_reg_inp,
                                                 aux_input_offset));
                    }

                    for (int ii = 0; ii < nb_oc_block; ii++) {
                        int aux_kernel_offset = kernel_offset(ii, cc, ic, ki);
                        if (jj_end - jj_start > 0) {
                            if (jcp.is_depthwise)
                                vpmovsxbd(
                                        zmm_wei, EVEX_compress_addr(aux_reg_ker,
                                                         aux_kernel_offset));
                            else
                                vmovups(zmm_wei, EVEX_compress_addr(aux_reg_ker,
                                                         aux_kernel_offset));
                        }
                        for (int jj = jj_start; jj < jj_end; jj++) {
                            compute(zmm_out(jj, ii), zmm_wei,
                                                     zmm_inp(jj, nb_oc_block));
                        }
                    }
                }
            }
        }
        add(aux_reg_ker, shift_kernel_ptr);
        add(aux_reg_inp, shift_input_ptr);
        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }
    L(skip_kh_loop);

    store_output(ur_w);

}

void jit_avx512_core_u8s8s32x_fwd_kernel::generate()
{
    int inp_shift_pad = jcp.typesize_in * (jcp.ur_w * jcp.stride_w - jcp.l_pad)
        * jcp.ic * jcp.ngroups;
    int inp_shift = jcp.typesize_in *
                        (jcp.ur_w * jcp.stride_w * jcp.ic * jcp.ngroups);
    int out_shift = jcp.typesize_out *
                        (jcp.ur_w * jcp.oc * jcp.ngroups);
    int acc_shift = jcp.typesize_acc *
                        (jcp.ur_w * jcp.oc_block * jcp.nb_oc_blocking);

    preamble();

    xor_(reg_scratch, reg_scratch);
    Reg16 _t = reg_scratch.cvt16();
    mov(_t, 0x1);
    vpbroadcastw(zmm_one, _t);

    mov(reg_inp, ptr[param1 + GET_OFF(src)]);
    mov(reg_out, ptr[param1 + GET_OFF(dst)]);
    mov(reg_ker, ptr[param1 + GET_OFF(filt)]);
    mov(reg_kh, ptr[param1 + GET_OFF(kh_padding)]);
    mov(reg_acc_s32, ptr[param1 + GET_OFF(acc_s32)]);

    int r_pad = nstl::max(0, (jcp.ow - 1) * jcp.stride_w
                    + (jcp.kw - 1) * (jcp.dilate_w + 1)
                    - (jcp.iw + jcp.l_pad - 1));
    int n_oi = jcp.ow / jcp.ur_w;
    int r_pad1 = (jcp.ur_w * n_oi - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1);
    if (r_pad1 > 0) n_oi--;

    xor_(reg_oi, reg_oi);
    if (jcp.ow == jcp.ur_w) {
        compute_loop(jcp.ur_w, jcp.l_pad, r_pad);
    } else {
        if (n_oi == 0) {
            compute_loop(jcp.ur_w, jcp.l_pad, r_pad1);
            add(reg_inp, inp_shift_pad);
            add(reg_out, out_shift);
            add(reg_acc_s32, acc_shift);
            if (jcp.ur_w_tail != 0) {
                compute_loop(jcp.ur_w_tail, 0, r_pad);
            }
        } else {
            if (jcp.l_pad > 0) {
                compute_loop(jcp.ur_w, jcp.l_pad, 0);
                add(reg_inp, inp_shift_pad);
                add(reg_out, out_shift);
                add(reg_acc_s32, acc_shift);

                inc(reg_oi);
            }
            if ((jcp.l_pad <= 0 && n_oi > 0) || (jcp.l_pad > 0 && n_oi > 1)) {
                if (jcp.l_pad <= 0 && r_pad1 > 0)
                    n_oi--;
                Label ow_loop_label;
                L(ow_loop_label); {
                    compute_loop(jcp.ur_w, 0, 0);
                    add(reg_inp, inp_shift);
                    add(reg_out, out_shift);
                    add(reg_acc_s32, acc_shift);

                    inc(reg_oi);
                    cmp(reg_oi, n_oi);
                    jl(ow_loop_label, T_NEAR);
                }
            }
            if (r_pad1 > 0) {
                compute_loop(jcp.ur_w, 0, r_pad1);
                add(reg_inp, inp_shift);
                add(reg_out, out_shift);
                add(reg_acc_s32, acc_shift);
            }
            if (jcp.ur_w_tail != 0) {
                compute_loop(jcp.ur_w_tail, 0, r_pad);
            }
        }
    }

    postamble();
}

bool jit_avx512_core_u8s8s32x_fwd_kernel::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr)
{
    using namespace primitive_kind;
    const auto &p = attr.post_ops_;

    auto is_relu = [&](int idx) {
        return p.entry_[idx].kind == eltwise
            && p.entry_[idx].eltwise.scale == 1.
            && p.entry_[idx].eltwise.alg == alg_kind::eltwise_relu
            && p.entry_[idx].eltwise.alpha == 0.;
    };

   switch (p.len_) {
    case 0: return true;
    case 1: return true
                && implication(jcp.with_eltwise, p.contain(sum, 0))
                && implication(!jcp.with_eltwise, is_relu(0) || p.contain(sum, 0));
    case 2: return true
                && implication(jcp.with_eltwise, p.contain(sum, 0) && is_relu(1))
                && implication(!jcp.with_eltwise, false
                        || (p.contain(sum, 0) && is_relu(1))
                        || (p.contain(sum, 1) && is_relu(0)));
    case 3: return true
                && jcp.with_eltwise == false
                && (is_relu(0) && p.contain(sum, 1) && is_relu(2));
    default: return false;
    }

    return false;
}

status_t jit_avx512_core_u8s8s32x_fwd_kernel::init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, cpu_memory_t::pd_t &src_pd,
            cpu_memory_t::pd_t &weights_pd, cpu_memory_t::pd_t &dst_pd,
            cpu_memory_t::pd_t &bias_pd, const primitive_attr_t &attr,
            bool with_relu, float relu_negative_slope)
{
    using namespace prop_kind;

    const memory_desc_wrapper src_d(&src_pd);
    const memory_desc_wrapper weights_d(&weights_pd);
    const memory_desc_wrapper dst_d(&dst_pd);
    const memory_desc_wrapper bias_d(&bias_pd);

    const int regs = 28;
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;

    if (!(mayiuse(avx512_core) &&
            src_d.data_type() == data_type::u8
         && weights_d.data_type() == data_type::s8
         && one_of(dst_d.data_type(), data_type::f32, data_type::s32,
            data_type::s8, data_type::u8)))
        return status::unimplemented;

    jcp = zero<decltype(jcp)>();
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = dst_d.dims()[2];
    jcp.ow = dst_d.dims()[3];
    jcp.kh = weights_d.dims()[with_groups + 2];
    jcp.kw = weights_d.dims()[with_groups + 3];
    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];
    jcp.src_fmt = src_d.format();
    jcp.with_bias = cd.bias_desc.format != memory_format::undef;
    jcp.with_eltwise = with_relu;
    jcp.eltwise_alpha = relu_negative_slope;
    jcp.ur_h = 1;

    if (!implication(with_relu, relu_negative_slope == 0.))
        return status::unimplemented;

    jcp.is_depthwise = true && with_groups && everyone_is(1, jcp.ic, jcp.oc);
    if (jcp.is_depthwise) {
        jcp.ch_block = 16;
        jcp.ic_block = 1;
        jcp.oc_block = 1;
        if (jcp.ngroups % jcp.ch_block != 0)
            return status::unimplemented;
    } else {
        jcp.ch_block = 1;
        jcp.ic_block = 16;
        jcp.oc_block = 16;
        if (jcp.ic % jcp.ic_block != 0)
            return status::unimplemented;
    }

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    jcp.ver = ver_avx512_core;
    if (mayiuse(avx512_core_vnni))
        jcp.ver = ver_vnni;

    const auto w_format = with_groups
        ? (jcp.is_depthwise ? Goihw16g : gOIhw4i16o4i) : OIhw4i16o4i;
    if (weights_d.format() == any)
        CHECK(weights_pd.set_format(w_format));
    if (weights_d.format() != w_format)
        return status::unimplemented;

    if (dst_d.format() == any)
        CHECK(dst_pd.set_format(nhwc));
    if (dst_d.format() != nhwc)
        return status::unimplemented;
    if (src_d.format() == any)
        CHECK(src_pd.set_format(nhwc));
    if (src_d.format() != nhwc)
        return status::unimplemented;
    if (jcp.with_bias) {
        if (bias_d.format() == any)
            CHECK(bias_pd.set_format(x));
        if (bias_d.format() != x)
            return status::unimplemented;
    }

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_acc = sizeof(int32_t);
    jcp.typesize_bia = jcp.with_bias
        ? types::data_type_size(bias_d.data_type())
        : 0;

    jcp.nb_ch = jcp.ngroups / jcp.ch_block;
    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    jcp.nb_ic_blocking = (!(jcp.nb_ic % 8))
                        ? 8
                        : (!(jcp.nb_ic % 4))
                            ? 4
                            : (!(jcp.nb_ic % 2)) ? 2 : 1;
    if (jcp.kh >= 7 || jcp.kw >= 7) // Note: Large code issue on SKX
        jcp.nb_ic_blocking = (!(jcp.nb_ic % 4))
                            ? 4
                            : (!(jcp.nb_ic % 2)) ? 2 : 1;

    // If OC blocking is incommensurate with the number of OC blocks (general
    // requirement for all convolutions), or if it results in an unrolling
    // factor smaller than the left padding (special requirement for SSD:fc6),
    // then search for a smaller OC blocking that satisfies both constraints.
    jcp.nb_oc_blocking = nstl::min(4, jcp.nb_oc);
    for (; jcp.nb_oc_blocking > 1; jcp.nb_oc_blocking--)
        if (jcp.nb_oc % jcp.nb_oc_blocking == 0
                && jcp.l_pad <= regs / (jcp.nb_oc_blocking + 1))
            break;

    jcp.ur_w = regs / (jcp.nb_oc_blocking + 1);
    if (jcp.ow < jcp.ur_w)  jcp.ur_w = jcp.ow;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    bool args_ok = true
        && jcp.oc % jcp.oc_block == 0
        && jcp.l_pad <= jcp.ur_w
        && implication(!jcp.is_1stconv, jcp.ic % jcp.ic_block == 0);
    if (!args_ok)
        return status::unimplemented;

    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
                    + (jcp.kw - 1) * (jcp.dilate_w + 1)
                    - (jcp.iw + jcp.l_pad - 1));
    if (r_pad_no_tail > jcp.ur_w)
        return status::unimplemented;

    pick_loop_order(jcp);

    jcp.nb_ic_L2 = jcp.nb_ic;

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    assert(utils::implication(!jcp.is_oc_scale, oscales.mask_ == 0));

    return status::success;
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
