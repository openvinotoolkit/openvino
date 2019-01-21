/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include "jit_uni_x8s8s32x_1x1_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_1x1_conv_call_s, field)

#include <iostream>

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::types;

using namespace Xbyak;

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_1x1_conv_fwd_kernel<isa>::cvt2ps(data_type_t type_in,
        Vmm vmm_in, const Xbyak::Operand &op) {
    switch (type_in) {
    case data_type::f32:
    case data_type::s32: vmovups(vmm_in, op); break;
    case data_type::s8: vpmovsxbd(vmm_in, op); break;
    case data_type::u8: vpmovzxbd(vmm_in, op); break;
    default: assert(!"unsupported data type");
    }
    if (type_in != data_type::f32)
        vcvtdq2ps(vmm_in, vmm_in);
}

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_1x1_conv_fwd_kernel<isa>::loop_os(int oc_loop_blk)
{
    mov(aux_reg_dst_data, reg_dst_data);

    Label loop_os;
    Label loop_ow_tail;

    mov(reg_ow_loop_work, jcp.ow);

    L(loop_os); {
        assert(jcp.os_block == jcp.ur);
        cmp(reg_ow_loop_work, jcp.ow_tail);
        je(loop_ow_tail, T_NEAR);

        ic_loop(oc_loop_blk, jcp.ur);

        sub(reg_ow_loop_work, jcp.ur);

        add(reg_src_data, jcp.os_loop_src_step);
        add(aux_reg_dst_data, jcp.os_loop_dst_step);

        sub(reg_loop_os_iter, jcp.os_block);
        cmp(reg_loop_os_iter, jcp.os_block);
        jge(loop_os, T_NEAR);

        L(loop_ow_tail); {
            if (jcp.ow_tail > 0) {
                ic_loop(oc_loop_blk, jcp.ow_tail);
            }

            add(reg_src_data, jcp.os_loop_src_tail_step);
            add(aux_reg_dst_data, jcp.os_loop_dst_tail_step);

            mov(reg_ow_loop_work, jcp.ow);

            sub(reg_loop_os_iter, jcp.ow_tail);
            cmp(reg_loop_os_iter, 0);
            jg(loop_os, T_NEAR);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_1x1_conv_fwd_kernel<isa>::ic_loop(int oc_loop_blk, int ur)
{
    auto vreg_wei = [=](int i) {
        return Vmm(ur * oc_loop_blk + i);
    };

    auto vreg_accum_vmm = [=](int i, int j) {
        return Vmm(j * oc_loop_blk + i);
    };

    auto vreg_accum_xmm = [=](int i, int j) {
        return Xmm(j * oc_loop_blk + i);
    };

    auto src_ptr = [=](int u, int j) {
        size_t offt = j * jcp.ic * jcp.stride_w + u*jcp.ic_block;
        return ptr[aux_reg_src_data + jcp.typesize_in * offt];
    };

    auto wei_ptr = [=](int u, int i) {
        size_t offt = i*jcp.nb_ic*jcp.oc_block*jcp.ic_block + u*jcp.ic_block * jcp.oc_block;
        return ptr[aux_reg_weight_data + offt * jcp.typesize_in];
    };

    auto output_ptr = [=](int i, int j) {
        return ptr[aux_reg_dst_data + (i * jcp.oc_block + j * jcp.oc) *
                                              jcp.typesize_out];
    };

    auto init = [&]() {
        for (int i = 0; i < oc_loop_blk; ++i) {
            for (int j = 0; j < ur; ++j) {
                auto vmm_acc = vreg_accum_vmm(i, j);
                uni_vpxor(vmm_acc, vmm_acc, vmm_acc);
            }
        }

        for (int i = 0; i < oc_loop_blk; ++i)
            uni_vmovdqu(vreg_wei(i), wei_ptr(0, i));

        uni_vpbroadcastd(vreg_src, src_ptr(0, 0));
    };

    auto store = [=]() {
        mov(reg_scales, ptr[this->param1 + GET_OFF(scales)]);
        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        for (int j = 0; j < ur; ++j)
            for (int i = 0; i < oc_loop_blk; ++i) {
                int b_off = i*jcp.oc_block;

                if (jcp.with_bias) {
                    switch (jcp.bia_dt) {
                        case data_type::f32:
                        case data_type::s32: vmovups(vmm_bias, ptr[reg_bias_data + b_off*jcp.typesize_bia]); break;
                        case data_type::s8: vpmovsxbd(vmm_bias, ptr[reg_bias_data + b_off*jcp.typesize_bia]); break;
                        case data_type::u8: vpmovzxbd(vmm_bias, ptr[reg_bias_data + b_off*jcp.typesize_bia]); break;
                        default: assert(!"unsupported dst data type");
                    }
                }
                if (jcp.bia_dt != data_type::f32)
                    vcvtdq2ps(vmm_bias, vmm_bias);

                Vmm vmm_dst = vreg_accum_vmm(i, j);
                Xmm xmm_dst = vreg_accum_xmm(i, j);

                vcvtdq2ps(vmm_dst, vmm_dst);

                if (jcp.with_bias)
                    vaddps(vmm_dst, vmm_dst, vmm_bias);

                int s_off = jcp.is_oc_scale * (sizeof(float) * (i*jcp.oc_block));
                vmulps(vmm_dst, vmm_dst, ptr[reg_scales + s_off]);

                if (jcp.with_sum) {
                    Ymm vmm_prev_dst = Ymm(12);
                    cvt2ps(jcp.dst_dt, vmm_prev_dst, output_ptr(i, j));
                    vaddps(vmm_dst, vmm_prev_dst);
                }

                if (maybe_relu(0))
                    vmaxps(vmm_dst, vmm_zero, vmm_dst);

                if (maybe_relu(1))
                    vmaxps(vmm_dst, vmm_zero, vmm_dst);

                if (jcp.dst_dt != data_type::f32) {
                    if (attr_.round_mode_ == round_mode::nearest)
                        if (isa == avx512_common) {
                            vcvtps2dq(vmm_dst | T_rn_sae, vmm_dst);
                        } else {
                            vcvtps2dq(vmm_dst, vmm_dst);
                        }
                    else if (attr_.round_mode_ == round_mode::down) {
                        if (isa == avx512_common) {
                            vcvtps2dq(vmm_dst | T_rd_sae, vmm_dst);
                        } else {
                            vroundps(vmm_dst, vmm_dst, 1);
                            vcvtps2dq(vmm_dst, vmm_dst);
                        }
                    } else
                        assert(!"unimplemented");
                }

                switch (jcp.dst_dt) {
                    case data_type::f32:
                    case data_type::s32: vmovups(output_ptr(i, j), vmm_dst); break;
                    case data_type::s8:
                        if (isa == avx512_common) {
                            vpmovsdb(xmm_dst, vmm_dst);
                            vmovups(output_ptr(i, j), xmm_dst);
                        } else if (isa == avx2) {
                            Ymm ymm_dst = Ymm(vmm_dst.getIdx());

                            vpackssdw(ymm_dst, ymm_dst, ymm_dst);
                            vpermq(ymm_dst, ymm_dst, 0x08);
                            vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                            vmovq(output_ptr(i, j), xmm_dst);
                        }
                        break;
                    case data_type::u8:
                        if (isa == avx512_common) {
                            vpmovusdb(xmm_dst, vmm_dst);
                            vmovups(output_ptr(i, j), xmm_dst);
                        } else if (isa == avx2) {
                            Ymm ymm_dst = Ymm(vmm_dst.getIdx());

                            vpackusdw(ymm_dst, ymm_dst, ymm_dst);
                            vpermq(ymm_dst, ymm_dst, 0x08);
                            vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                            vmovq(output_ptr(i, j), xmm_dst);
                        }
                        break;
                    default: assert(!"unknown dst_dt");
                }
            }
    };

    auto fma_block = [=]() {
        for (int j = 0; j < ur; ++j) {
            for (int i = 0; i < oc_loop_blk; i++) {
                vpmaddubsw(vreg_sum_0, vreg_src, vreg_wei(i));
                vpmaddwd(vreg_sum_0, vreg_sum_0, vmm_one);
                vpaddd(vreg_accum_vmm(i, j), vreg_accum_vmm(i, j), vreg_sum_0);

                if (j == ur - 1) {
                    uni_vmovdqu(vreg_wei(i), wei_ptr(1, i));
                }
            }

            if (j < ur - 1)
                uni_vpbroadcastd(vreg_src, src_ptr(0, j + 1));
        }

        uni_vpbroadcastd(vreg_src, src_ptr(1, 0));
    };

    mov(aux_reg_weight_data, reg_weight_data);
    mov(aux_reg_src_data, reg_src_data);

    init();

    Label ic_loop;
    Label exit;

    xor_(reg_loop_ic_iter, reg_loop_ic_iter);
    L(ic_loop); {
        cmp(reg_loop_ic_iter, jcp.nb_ic);
        jge(exit, T_NEAR);

        fma_block();

        add(aux_reg_src_data, jcp.ic_block * jcp.typesize_in);
        add(aux_reg_weight_data, jcp.ic_block * jcp.oc_block * jcp.typesize_in);
        inc(reg_loop_ic_iter);
        jmp(ic_loop, T_NEAR);
    }

    L(exit);

    store();
}

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_1x1_conv_fwd_kernel<isa>::generate()
{
    preamble();

    mov(reg_scratch, 0x1);
    movq(xmm_one, reg_scratch);
    vpbroadcastw(vmm_one, xmm_one);

    mov(reg_weight_data, ptr[param1 + GET_OFF(oc_data)]);
    mov(reg_dst_data,    ptr[param1 + GET_OFF(output_data)]);
    if (jcp.with_bias) {
        mov(reg_bias_data, ptr[param1 + GET_OFF(bias_data)]);
    }

    mov(reg_oc_loop_work, ptr[param1 + GET_OFF(oc_dim)]);
    mov(reg_src_data, ptr[param1 + GET_OFF(is_data)]);
    mov(reg_loop_os_iter,  ptr[param1 + GET_OFF(os_dim)]);

    Label oc_blocks_tail_label;
    Label exit_label;

    int oc_blocks_tail = jcp.nb_oc % jcp.nb_oc_blocking;

    cmp(reg_oc_loop_work, jcp.nb_oc_blocking);
    jne(oc_blocks_tail ? oc_blocks_tail_label : exit_label, T_NEAR);

    loop_os(jcp.nb_oc_blocking); // channel main loop
    jmp(exit_label, T_NEAR);

    if (oc_blocks_tail) {
        L(oc_blocks_tail_label);

        cmp(reg_oc_loop_work, oc_blocks_tail);
        jne(exit_label, T_NEAR);

        loop_os(oc_blocks_tail); // channel tail loop
    }

    L(exit_label);

    postamble();
}

template <cpu_isa_t isa>
bool jit_uni_x8s8s32x_1x1_conv_fwd_kernel<isa>::post_ops_ok(
        jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    auto is_relu = [&](int idx) { return p.entry_[idx].is_relu(); };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(); };

    switch (p.len_) {
        case 0: return true; // no post_ops
        case 1: return !jcp.with_eltwise && (is_relu(0) || is_sum(0)); // sum OR relu
        case 2: return !jcp.with_eltwise && (is_sum(0) && is_relu(1)); // sum->relu
        default: return false;
    }

    return false;
}

template <cpu_isa_t isa>
bool jit_uni_x8s8s32x_1x1_conv_fwd_kernel<isa>::maybe_relu(int position) {
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

template <cpu_isa_t isa>
status_t jit_uni_x8s8s32x_1x1_conv_fwd_kernel<isa>::init_conf(jit_1x1_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const memory_desc_wrapper &bias_pd, const primitive_attr_t &attr,
        bool with_relu, float relu_negative_slope)
{
    if (!mayiuse(isa)) return status::unimplemented;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;

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

    jcp.with_bias = cd.bias_desc.format != memory_format::undef;
    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;

    jcp.src_fmt = src_d.format();
    jcp.with_eltwise = with_relu;
    jcp.eltwise_alpha = relu_negative_slope;

    jcp.os = jcp.oh * jcp.ow;
    jcp.is = jcp.ih * jcp.iw;

    auto desired_wei_fmt = OhIw8o4i;
    auto desired_gr_wei_fmt = gOhIw8o4i;

    int simd_w = isa == avx512_common ? 16 : 8;

    bool args_ok = true
        && jcp.ngroups == 1
        && src_d.format() == nhwc
        && one_of(weights_d.format(), desired_wei_fmt, desired_gr_wei_fmt)
        && one_of(cd.bias_desc.format, memory_format::undef, any, x)
        && dst_d.format() == nhwc
        && jcp.oc % simd_w == 0 && jcp.ic % simd_w == 0
        && jcp.t_pad == 0 && jcp.l_pad == 0
        && jcp.kh == 1 && jcp.kw == 1
        && jcp.stride_h == 1 && jcp.stride_w == 1;

    if (!args_ok) return status::unimplemented;

    jcp.ic_block = 4;
    jcp.oc_block = simd_w;

    jcp.ur = 2;
    jcp.ow_tail = jcp.ow % jcp.ur;

    int oc_blocking{ 0 };
    int oc_blocking_max{ 0 };
    int os_blocking{ 0 };
    int os_blocking_max{ 0 };
    int ic_blocking{ 0 };

    jcp.ic_dim = jcp.ic;
    jcp.oc_dim = jcp.oc;
    jcp.is_dim = jcp.is;
    jcp.os_block = jcp.ur;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_acc = sizeof(int32_t);
    jcp.typesize_bia = jcp.with_bias
                       ? types::data_type_size(bias_pd.data_type())
                       : 0;

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;

    assert(IMPLICATION(!jcp.is_oc_scale, oscales.mask_ == 0));

    jcp.ic_loop_src_step = jcp.ic_block * jcp.ic_loop_unroll * jcp.typesize_in;
    jcp.ic_loop_wei_step = jcp.ic_block * jcp.ic_loop_unroll * jcp.oc_block * jcp.typesize_in;

    jcp.os_loop_dst_step = jcp.ur * jcp.oc * jcp.typesize_out;
    jcp.os_loop_acc_step = jcp.ur * jcp.oc_block * jcp.typesize_acc;
    jcp.os_loop_src_step = jcp.stride_w * jcp.ur * jcp.ic * jcp.typesize_in;
    jcp.os_loop_dst_tail_step = jcp.ow_tail * jcp.oc * jcp.typesize_out;
    jcp.os_loop_acc_tail_step = jcp.ow_tail * jcp.oc_block * jcp.typesize_acc;
    jcp.os_loop_src_tail_step = jcp.stride_w * jcp.ow_tail * jcp.ic * jcp.typesize_in
             + ((jcp.stride_h-1)*jcp.iw*jcp.ic*jcp.typesize_in);

    oc_blocking     = 4 * jcp.oc_block;
    oc_blocking_max = 4 * jcp.oc_block;
    os_blocking     = 48; // affects oc balancing across threads
    os_blocking_max = 320;
    ic_blocking     = 4*128; // affects L1$ utilization

    assert(oc_blocking);
    assert(oc_blocking_max);
    assert(os_blocking);
    assert(os_blocking_max);
    assert(ic_blocking);

    assert(jcp.os_block % jcp.ur == 0);
    jcp.ur_tail = jcp.is_dim % jcp.ur;

    jcp.nb_oh_blocking     = nstl::max(1, os_blocking     / jcp.ow);
    jcp.nb_oh_blocking_max = nstl::max(1, os_blocking_max / jcp.ow);
    jcp.nb_oc_blocking     = oc_blocking / jcp.oc_block;
    jcp.nb_oc_blocking_max = oc_blocking_max / jcp.oc_block;
    jcp.nb_ic_blocking     = ic_blocking / jcp.ic_block;

    jcp.nb_oc = div_up(jcp.oc_dim, jcp.oc_block);

    jcp.nb_ic = jcp.ic / jcp.ic_block;

    return status::success;
}

template struct jit_uni_x8s8s32x_1x1_conv_fwd_kernel<avx2>;
template struct jit_uni_x8s8s32x_1x1_conv_fwd_kernel<sse42>;

}
}
}
