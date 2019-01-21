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

#include "jit_uni_x8s8s32x_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

using namespace Xbyak;

template <cpu_isa_t isa>
bool jit_uni_x8s8s32x_conv_fwd_kernel<isa>::maybe_relu(int position) {
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
void jit_uni_x8s8s32x_conv_fwd_kernel<isa>::cvt2ps(data_type_t type_in, Vmm vmm_in,
        const Xbyak::Operand &op, bool scalar_load) {
    Xmm xmm_in = Xmm(vmm_in.getIdx());

    switch (type_in) {
        case data_type::f32:
        case data_type::s32:
            if (scalar_load) {
                movsd(xmm_in, op);
            } else {
                uni_vmovups(vmm_in, op);
            }
            break;
        case data_type::s8:
            if (scalar_load) {
                movsx(reg_tmp_32, op);
                movq(xmm_in, reg_tmp_64);
            } else {
                uni_vpmovsxbd(vmm_in, op);
            }
            break;
        case data_type::u8:
            if (scalar_load) {
                movzx(reg_tmp_32, op);
                movq(xmm_in, reg_tmp_64);
            } else {
                uni_vpmovzxbd(vmm_in, op);
            }
            break;
        default: assert(!"unsupported data type");
    }

    if (type_in != data_type::f32)
        uni_vcvtdq2ps(vmm_in, vmm_in);
}

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_conv_fwd_kernel<isa>::store_dst(const Xbyak::Address &op, Vmm vmm_dst, bool scalar_store) {
    Ymm ymm_dst = Ymm(vmm_dst.getIdx());
    Xmm xmm_dst = Xmm(vmm_dst.getIdx());

    switch (jcp.dst_dt) {
        case data_type::f32:
        case data_type::s32:
            if (scalar_store) {
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_32);
            } else {
                uni_vmovups(op, vmm_dst);
            }
            break;
        case data_type::s8:
            uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);

            if (isa != sse42 && !scalar_store)
                vpermq(ymm_dst, ymm_dst, 0x08);

            uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);

            if (scalar_store) {
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
            } else {
                if (isa != sse42)
                    vmovq(op, xmm_dst);
                else
                    movd(op, xmm_dst);
            }
            break;
        case data_type::u8:
            uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);

            if (isa != sse42 && !scalar_store)
                vpermq(ymm_dst, ymm_dst, 0x08);

            uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);

            if (scalar_store) {
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
            } else {
                if (isa != sse42)
                    vmovq(op, xmm_dst);
                else
                    movd(op, xmm_dst);
            }

            break;
        default:
            assert(!"unknown dst_dt");
    }
}

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_conv_fwd_kernel<isa>::apply_filter(int ur_w, int pad_l, int pad_r, int oc_blocks, int oc_step,
        int tail_size, bool h_padded) {
    int kw = jcp.kw;
    int kh = jcp.kh;
    int nb_ic = jcp.nb_ic;
    int stride_w = jcp.stride_w;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;

    int repeats = isa == sse42 && oc_step > (oc_blk / 2) ? 2 : 1;

    for (int ki = 0; ki < kw; ki++) {
        int jj_start = nstl::max(0, div_up(pad_l - ki * dilate_w, stride_w));
        int jj_end = ur_w - nstl::max(0, div_up(ki*dilate_w+pad_r-(kw-1)*dilate_w, stride_w));

        int _start = (jcp.signed_input) ? 0 : jj_start;
        int _end = (jcp.signed_input) ? ur_w : jj_end;

        for (int r = 0; r < repeats; r++) {
            for (int jj = _start; jj < _end; jj++) {
                int inp_off = (ki * dilate_w + jj * stride_w - pad_l) * jcp.ic * jcp.ngroups;
                    if (tail_size > 0) {
                        if (h_padded || jj < jj_start || jj >= jj_end) {
                            uni_vpxor(get_src_reg(jj), get_src_reg(jj), get_src_reg(jj));
                            uni_vpsubb(get_src_reg(jj), get_src_reg(jj), vmm_shift);
                            uni_vandps(get_src_reg(jj), get_src_reg(jj), vmm_mask);
                            uni_vpbroadcastd(get_src_reg(jj), Xmm(get_src_reg(jj).getIdx()));
                        } else {
                            uni_vpbroadcastd(get_src_reg(jj), ptr[aux1_reg_input + jcp.typesize_in * inp_off]);

                            if (jcp.signed_input) {
                                uni_vpsubb(get_src_reg(jj), get_src_reg(jj), vmm_shift);
                            }

                            uni_vandps(get_src_reg(jj), get_src_reg(jj), vmm_mask);
                            uni_vpbroadcastd(get_src_reg(jj), Xmm(get_src_reg(jj).getIdx()));
                        }
                    } else {
                        if (h_padded || jj < jj_start || jj >= jj_end) {
                            uni_vpxor(get_src_reg(jj), get_src_reg(jj), get_src_reg(jj));
                        } else {
                            uni_vpbroadcastd(get_src_reg(jj), ptr[aux1_reg_input + jcp.typesize_in * inp_off]);
                        }

                        if (jcp.signed_input)
                            uni_vpsubb(get_src_reg(jj), get_src_reg(jj), vmm_shift);
                    }
            }

            for (int ii = 0; ii < oc_blocks; ii++) {
                int ker_off = ii * nb_ic * kh * kw * ic_blk * oc_blk + ki * ic_blk * oc_blk + r * ic_blk * (oc_blk / 2);
                uni_vmovups(get_ker_reg(0), ptr[aux1_reg_kernel + jcp.typesize_in * ker_off]);

                for (int jj = _start; jj < _end; jj++) {
                    Vmm vmm_src = get_src_reg(jj);
                    if (isa == sse42) {
                        uni_vmovups(get_tmp_reg(0), vmm_src);
                        uni_vpmaddubsw(get_tmp_reg(0), get_tmp_reg(0), get_ker_reg(0));
                    } else {
                        uni_vpmaddubsw(get_tmp_reg(0), vmm_src, get_ker_reg(0));
                    }
                    uni_vpmaddwd(get_tmp_reg(0), get_tmp_reg(0), vmm_one);
                    uni_vpaddd(get_acc_reg(r*jcp.ur_w*jcp.nb_oc_blocking + ur_w * ii + jj),
                               get_acc_reg(r*jcp.ur_w*jcp.nb_oc_blocking + ur_w * ii + jj), get_tmp_reg(0));
                }
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_conv_fwd_kernel<isa>::oh_step_unroll_kw(int ur_w,
        int pad_l, int pad_r, int oc_blocks, int oc_step, bool h_padded) {
    int kw = jcp.kw;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;

    Label ic_main_loop;
    Label ic_tail;
    Label exit;

    mov(aux1_reg_input, aux_reg_input);
    mov(aux1_reg_kernel, aux_reg_kernel);

    mov(reg_ic_iter, jcp.ic);

    L(ic_main_loop); {
        cmp(reg_ic_iter, ic_blk);
        jl(ic_tail, T_NEAR);

        apply_filter(ur_w, pad_l, pad_r, oc_blocks, oc_step, 0, h_padded);

        add(aux1_reg_input, ic_blk * jcp.typesize_in);
        add(aux1_reg_kernel, kw * ic_blk * oc_blk * jcp.typesize_in);
        sub(reg_ic_iter, ic_blk);
        jmp(ic_main_loop, T_NEAR);
    }

    L(ic_tail);
    int ic_tail_size = jcp.ic % jcp.ic_block;

    if (ic_tail_size > 0)
        apply_filter(ur_w, pad_l, pad_r, oc_blocks, oc_step, ic_tail_size, h_padded);

    L(exit);
}

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_conv_fwd_kernel<isa>::kh_loop(int ur_w, int pad_l, int pad_r, int oc_blocks, int oc_step) {
    int iw = jcp.iw;
    int kw = jcp.kw;
    int dilate_h = jcp.dilate_h + 1;
    const int inp_mult = jcp.ic * dilate_h * jcp.ngroups;

    Label t_overflow_label, no_t_overflow_label,
          b_overflow_label, no_b_overflow_label;

    mov(aux_reg_input, reg_input);
    mov(aux_reg_kernel, reg_kernel);

    mov(imm_addr64, l_table);
    uni_vmovups(vmm_one,   ptr[imm_addr64 + 0 * vlen]);
    uni_vmovups(vmm_shift, ptr[imm_addr64 + 1 * vlen]);
    uni_vmovups(vmm_mask, ptr[imm_addr64 + 4 * vlen]);

    if (jcp.signed_input) {
        mov(reg_overflow,  ptr[param1 + GET_OFF(t_overflow)]);
        cmp(reg_overflow, 0);
        je(no_t_overflow_label, T_NEAR);
        L(t_overflow_label); {
            oh_step_unroll_kw(ur_w, pad_l, pad_r, oc_blocks, oc_step, true);

            add(aux_reg_kernel, jcp.typesize_in * kw * jcp.oc_block * rnd_up(jcp.ic, jcp.ic_block));
            dec(reg_overflow);
            cmp(reg_overflow, 0);
            jg(t_overflow_label, T_NEAR);
        }
        L(no_t_overflow_label);
    }

    Label skip_kh_loop;
    mov(reg_kj, ptr[this->param1 + GET_OFF(kh_padding)]);
    if ((jcp.signed_input) || (!jcp.signed_input &&
                               (jcp.kh - 1) * (jcp.dilate_h + 1) < nstl::max(jcp.t_pad, jcp.b_pad))) {
        cmp(reg_kj, 0);
        je(skip_kh_loop, T_NEAR);
    }

    Label kh_label;
    L(kh_label);
    {
        oh_step_unroll_kw(ur_w, pad_l, pad_r, oc_blocks, oc_step, false);

        add(aux_reg_kernel, jcp.typesize_in * kw * jcp.oc_block * rnd_up(jcp.ic, jcp.ic_block));
        add(aux_reg_input, jcp.typesize_in * iw * inp_mult);

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    L(skip_kh_loop);

    if (jcp.signed_input) {
        mov(reg_overflow,  ptr[param1 + GET_OFF(b_overflow)]);
        cmp(reg_overflow, 0);
        je(no_b_overflow_label, T_NEAR);
        L(b_overflow_label); {
            oh_step_unroll_kw(ur_w, pad_l, pad_r, oc_blocks, oc_step, true);

            add(aux_reg_kernel, jcp.typesize_in * kw * jcp.oc_block * rnd_up(jcp.ic, jcp.ic_block));
            dec(reg_overflow);
            cmp(reg_overflow, 0);
            jg(b_overflow_label, T_NEAR);
        }
        L(no_b_overflow_label);
    }
}

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_conv_fwd_kernel<isa>::width_blk_step(int ur_w, int pad_l, int pad_r, int oc_blocks, int oc_step)
{
    int repeats = isa == sse42 && oc_step > (jcp.oc_block / 2) ? 2 : 1;

    for (int r = 0; r < repeats; r++)
        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++)
                uni_vpxor(get_acc_reg(r*jcp.ur_w*jcp.nb_oc_blocking + ur_w * ii + jj),
                          get_acc_reg(r*jcp.ur_w*jcp.nb_oc_blocking + ur_w * ii + jj),
                          get_acc_reg(r*jcp.ur_w*jcp.nb_oc_blocking + ur_w * ii + jj));

    kh_loop(ur_w, pad_l, pad_r, oc_blocks, oc_step);

    pop(reg_scales_base);

    mov(imm_addr64, l_table);
    uni_vmovups(vmm_bias_alpha, ptr[imm_addr64 + 2 * vlen]);

    const auto &p = attr_.post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    const float p_sum_scale = (sum_idx != -1) ? p.entry_[sum_idx].sum.scale : 1.f;

    for (int r = 0; r < repeats; r++) {
        int tail_size = isa == avx2 ? oc_step : nstl::min(jcp.oc_block / 2, oc_step - r * jcp.oc_block / 2);
        bool is_scalar_store = isa == avx2 ? tail_size < jcp.oc_block : tail_size < jcp.oc_block / 2;

        if (is_scalar_store) {
            for (int jj = 0; jj < ur_w; jj++) {
                Vmm vmm_dst = get_acc_reg(r * jcp.ur_w * jcp.nb_oc_blocking + jj);
                uni_vcvtdq2ps(vmm_dst, vmm_dst);
                uni_vmovups(vmm_reminder_dst, vmm_dst);

                for (int oc = 0; oc < tail_size; oc++) {
                    uni_vmovups(vmm_dst, vmm_reminder_dst);

                    if (jcp.with_bias) {
                        int b_off = r * (jcp.oc_block / 2) + oc;
                        cvt2ps(jcp.bia_dt, vmm_bias, ptr[reg_bias_base + b_off * jcp.typesize_bia], true);

                        if (jcp.signed_input)
                            uni_vmulps(vmm_bias, vmm_bias, vmm_bias_alpha);
                    }
                    if (jcp.signed_input) {
                        int c_off = r * (jcp.oc_block / 2) + oc;
                        cvt2ps(data_type::s32, vmm_comp, ptr[reg_compensation_base + c_off * sizeof(int32_t)], true);
                    }

                    if (jcp.signed_input)
                        uni_vaddps(vmm_dst, vmm_dst, vmm_comp);
                    if (jcp.with_bias)
                        uni_vaddps(vmm_dst, vmm_dst, vmm_bias);

                    int s_off = jcp.is_oc_scale * (r * (jcp.oc_block / 2) + oc);
                    cvt2ps(mkldnn_f32, vmm_scale, ptr[reg_scales_base + s_off * sizeof(float)], true);
                    uni_vmulps(vmm_dst, vmm_dst, vmm_scale);

                    int o_off = jj * jcp.oc * jcp.ngroups + r * (jcp.oc_block / 2) + oc;
                    if (jcp.with_sum) {
                        uni_vpxor(vmm_prev_dst, vmm_prev_dst, vmm_prev_dst);
                        cvt2ps(jcp.dst_dt, vmm_prev_dst, ptr[reg_output + o_off * jcp.typesize_out], true);

                        if (p_sum_scale == 1.f) {
                            uni_vaddps(vmm_dst, vmm_dst, vmm_prev_dst);
                        } else {
                            uni_vfmadd231ps(vmm_dst, vmm_prev_dst, ptr[imm_addr64 + 3 * vlen]);
                        }
                    }

                    if (maybe_relu(0)) {
                        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
                        uni_vmaxps(vmm_dst, vmm_dst, vmm_zero);
                    }

                    if (maybe_relu(1)) {
                        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
                        uni_vmaxps(vmm_dst, vmm_dst, vmm_zero);
                    }

                    if (jcp.dst_dt != data_type::f32) {
                        if (attr_.round_mode_ == round_mode::nearest)
                            uni_vcvtps2dq(vmm_dst, vmm_dst);
                        else if (attr_.round_mode_ == round_mode::down) {
                            uni_vroundps(vmm_dst, vmm_dst, 1);
                            uni_vcvtps2dq(vmm_dst, vmm_dst);
                        } else
                            assert(!"unimplemented");
                    }

                    store_dst(ptr[reg_output + o_off * jcp.typesize_out], vmm_dst, true);

                    if (isa == avx2) {
                        vperm2i128(ymm_tmp, ymm_reminder_dst, ymm_reminder_dst, 0x01);
                        vpalignr(ymm_reminder_dst, ymm_tmp, ymm_reminder_dst, jcp.typesize_out);
                    } else {
                        psrldq(vmm_reminder_dst, jcp.typesize_out);
                    }
                }
            }
        } else {
            for (int ii = 0; ii < oc_blocks; ii++) {
                if (jcp.with_bias) {
                    int b_off = ii * jcp.oc_block + r * (jcp.oc_block / 2);
                    cvt2ps(jcp.bia_dt, vmm_bias, ptr[reg_bias_base + b_off * jcp.typesize_bia], false);

                    if (jcp.signed_input)
                        uni_vmulps(vmm_bias, vmm_bias, vmm_bias_alpha);
                }

                for (int jj = 0; jj < ur_w; jj++) {
                    Vmm vmm_dst = get_acc_reg(r * jcp.ur_w * jcp.nb_oc_blocking + ur_w * ii + jj);
                    uni_vcvtdq2ps(vmm_dst, vmm_dst);

                    if (jcp.signed_input) {
                        int c_off = ii * jcp.oc_block + r * (jcp.oc_block / 2);
                        cvt2ps(data_type::s32, vmm_comp, ptr[reg_compensation_base + c_off * sizeof(int32_t)], false);
                    }

                    if (jcp.signed_input)
                        uni_vaddps(vmm_dst, vmm_dst, vmm_comp);
                    if (jcp.with_bias)
                        uni_vaddps(vmm_dst, vmm_dst, vmm_bias);

                    int s_off = jcp.is_oc_scale * (ii * jcp.oc_block + r * (jcp.oc_block / 2));
                    cvt2ps(mkldnn_f32, vmm_scale, ptr[reg_scales_base + s_off * sizeof(float)], false);
                    uni_vmulps(vmm_dst, vmm_dst, vmm_scale);

                    int o_off = ii * jcp.oc_block + jj * jcp.oc * jcp.ngroups + r * (jcp.oc_block / 2);
                    if (jcp.with_sum) {
                        cvt2ps(jcp.dst_dt, vmm_prev_dst, ptr[reg_output + o_off * jcp.typesize_out], false);

                        if (p_sum_scale == 1.f) {
                            uni_vaddps(vmm_dst, vmm_dst, vmm_prev_dst);
                        } else {
                            uni_vfmadd231ps(vmm_dst, vmm_prev_dst, ptr[imm_addr64 + 3 * vlen]);
                        }
                    }

                    if (maybe_relu(0)) {
                        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
                        uni_vmaxps(vmm_dst, vmm_dst, vmm_zero);
                    }

                    if (maybe_relu(1)) {
                        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
                        uni_vmaxps(vmm_dst, vmm_dst, vmm_zero);
                    }

                    if (jcp.dst_dt != data_type::f32) {
                        if (attr_.round_mode_ == round_mode::nearest)
                            uni_vcvtps2dq(vmm_dst, vmm_dst);
                        else if (attr_.round_mode_ == round_mode::down) {
                            uni_vroundps(vmm_dst, vmm_dst, 1);
                            uni_vcvtps2dq(vmm_dst, vmm_dst);
                        } else
                            assert(!"unimplemented");
                    }

                    store_dst(ptr[reg_output + o_off * jcp.typesize_out], vmm_dst, false);
                }
            }
        }
    }

    push(reg_scales_base);
}

template <cpu_isa_t isa>
inline void jit_uni_x8s8s32x_conv_fwd_kernel<isa>::solve_common(int oc_blocks, int oc_step)
{
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int n_oi = jcp.ow / ur_w;
    int iw = jcp.iw;
    int kw = jcp.kw;
    int dilate_w = jcp.dilate_w + 1;
    int str_w = jcp.stride_w;
    const int inp_mult = jcp.ic * jcp.ngroups;

    int l_pad = jcp.l_pad;
    int r_pad = nstl::max(0, (int(jcp.ow) - 1) * str_w + (kw - 1) * dilate_w
            - (iw + l_pad - 1));
    int r_pad1 = (ur_w * n_oi - 1) * str_w + (kw - 1) * dilate_w
            - (iw + l_pad - 1);
    if (r_pad1 > 0) n_oi--;

    mov(reg_input, reg_input_base);
    mov(reg_output, reg_output_base);
    mov(reg_kernel, reg_kernel_base);

    push(reg_input_base);
    push(reg_output_base);
    push(reg_kernel_base);
    push(reg_scales_base);

    if (l_pad > 0) {
        n_oi--;
        if (n_oi < 0 && r_pad1 > 0)
            width_blk_step(ur_w, l_pad, r_pad1, oc_blocks, oc_step); // "lrpad"
        else
            width_blk_step(ur_w, l_pad, 0, oc_blocks, oc_step); // "lpad"
        add(reg_input, jcp.typesize_in * (ur_w * str_w - l_pad) * inp_mult);
        add(reg_output, jcp.typesize_out * ur_w * jcp.oc * jcp.ngroups);
    }

    Label ow_loop_label;
    xor_(reg_oi_iter, reg_oi_iter);

    if (n_oi > 0) {
        L(ow_loop_label);

        width_blk_step(ur_w, 0, 0, oc_blocks, oc_step); // "middle"
        add(reg_input, jcp.typesize_in * ur_w * str_w * inp_mult);
        add(reg_output, jcp.typesize_out * ur_w * jcp.oc * jcp.ngroups);

        inc(reg_oi_iter);
        cmp(reg_oi_iter, n_oi);
        jl(ow_loop_label, T_NEAR);
    }

    if (r_pad1 > 0 && n_oi >=0) {
        width_blk_step(ur_w, 0, r_pad1, oc_blocks, oc_step); // "rpad"
        add(reg_input, jcp.typesize_in * ur_w * str_w * inp_mult);
        add(reg_output, jcp.typesize_out * ur_w * jcp.oc * jcp.ngroups);
    }

    if (ur_w_tail != 0)
        width_blk_step(ur_w_tail, 0, r_pad, oc_blocks, oc_step); // "tail"

    pop(reg_scales_base);
    pop(reg_kernel_base);
    pop(reg_output_base);
    pop(reg_input_base);
}

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_conv_fwd_kernel<isa>::generate()
{
    this->preamble();

    mov(reg_kernel_base, ptr[this->param1 + GET_OFF(filt)]);
    mov(reg_input_base, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_output_base, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_oc, ptr[this->param1 + GET_OFF(oc_work)]);
    if (jcp.with_bias)
        mov(reg_bias_base, ptr[this->param1 + GET_OFF(bias)]);
    mov(reg_scales_base, ptr[this->param1 + GET_OFF(scales)]);
    if (jcp.signed_input)
        mov(reg_compensation_base, ptr[param1 + GET_OFF(compensation)]);

    Label main_loop_label;
    Label tail_label;
    Label exit_label;

    cmp(reg_oc, jcp.nb_oc_blocking * jcp.oc_block);
    jne(main_loop_label, T_NEAR);

    solve_common(jcp.nb_oc_blocking, jcp.oc_block);

    sub(reg_oc, jcp.nb_oc_blocking * jcp.oc_block);

    jmp(exit_label, T_NEAR);

    L(main_loop_label); {
        cmp(reg_oc, jcp.oc_block);
        jl(tail_label, T_NEAR);

        solve_common(1, jcp.oc_block);

        sub(reg_oc, jcp.oc_block);
        add(reg_kernel_base, jcp.oc_block * jcp.nb_ic * jcp.kh * jcp.kw * jcp.ic_block * jcp.typesize_in);
        add(reg_output_base, jcp.oc_block * jcp.typesize_out);
        add(reg_bias_base, jcp.oc_block * jcp.typesize_bia);
        add(reg_scales_base, jcp.is_oc_scale * jcp.oc_block * sizeof(float));
        add(reg_compensation_base, jcp.oc_block * sizeof(int32_t));

        jmp(main_loop_label, T_NEAR);
    }

    L(tail_label);

    solve_common(1, jcp.oc % jcp.oc_block);

    L(exit_label);

    this->postamble();

    prepare_table();
}

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_conv_fwd_kernel<isa>::prepare_table() {
    const auto &p = attr_.post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    const float p_sum_scale = (sum_idx != -1) ? p.entry_[sum_idx].sum.scale : 1.f;

    const uint16_t cvals_one[] = {
        0x0001,
    };

    const int8_t cvals_shift[] = {
        -128,
    };

    const int32_t cvals_scale[] = {
        float2int(jcp.wei_adj_scale)
    };

    const int32_t cvals_sum_scale[] = {
        float2int(p_sum_scale)
    };

    align(64);
    L(l_table);
    for (size_t i = 0; i < sizeof(cvals_one) / sizeof(cvals_one[0]); ++i) {
        for (size_t d = 0; d < vlen / sizeof(uint16_t); ++d) {
            dw(cvals_one[i]);
        }
    }

    for (size_t i = 0; i < sizeof(cvals_shift) / sizeof(cvals_shift[0]); ++i) {
        for (size_t d = 0; d < vlen / sizeof(int8_t); ++d) {
            db(cvals_shift[i]);
        }
    }

    for (size_t i = 0; i < sizeof(cvals_scale) / sizeof(cvals_scale[0]); ++i) {
        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(cvals_scale[i]);
        }
    }

    for (size_t i = 0; i < sizeof(cvals_sum_scale) / sizeof(cvals_sum_scale[0]); ++i) {
        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(cvals_sum_scale[i]);
        }
    }

    for (size_t i = 0; i < sizeof(cvals_shift) / sizeof(cvals_shift[0]); ++i) {
        for (size_t d = 0; d < vlen / sizeof(int8_t); ++d) {
            if ((int)d < jcp.ic % jcp.ic_block)
                db(255);
            else
                db(0);
        }
    }
}

template <cpu_isa_t isa>
bool jit_uni_x8s8s32x_conv_fwd_kernel<isa>::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
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
                       && IMPLICATION(jcp.with_eltwise, p.contain(sum, 0))
                       && IMPLICATION(!jcp.with_eltwise, is_relu(0) || p.contain(sum, 0));
        case 2: return true
                       && IMPLICATION(jcp.with_eltwise, p.contain(sum, 0) && is_relu(1))
                       && IMPLICATION(!jcp.with_eltwise, false
                                                         || (p.contain(sum, 0) && is_relu(1))
                                                         || (p.contain(sum, 1) && is_relu(0)));
        case 3: return true
                       && jcp.with_eltwise == false
                       && (is_relu(0) && p.contain(sum, 1) && is_relu(2));
        default: return false;
    }

    return false;
}

template <cpu_isa_t isa>
status_t jit_uni_x8s8s32x_conv_fwd_kernel<isa>::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, cpu_memory_t::pd_t &src_pd,
        cpu_memory_t::pd_t &weights_pd, cpu_memory_t::pd_t &dst_pd,
        cpu_memory_t::pd_t &bias_pd,
        const primitive_attr_t &attr, bool with_relu, float relu_negative_slope)
{
    if (!mayiuse(isa)) return status::unimplemented;

    const memory_desc_wrapper src_d(&src_pd);
    const memory_desc_wrapper weights_d(&weights_pd);
    const memory_desc_wrapper dst_d(&dst_pd);
    const memory_desc_wrapper bias_d(&bias_pd);

    jcp.prop_kind = cd.prop_kind;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;

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

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];

    jcp.src_fmt = src_d.format();
    jcp.with_bias = cd.bias_desc.format != memory_format::undef;
    jcp.with_eltwise = with_relu;
    jcp.eltwise_alpha = relu_negative_slope;

    jcp.signed_input = src_d.data_type() == data_type::s8;

    const int simd_w = 8;

    jcp.ic_block = 4;
    jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);

    jcp.oc_block = simd_w;
    jcp.oc_padded = rnd_up(jcp.oc, jcp.oc_block);
    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    if (!jcp.with_eltwise) {
        jcp.with_eltwise = p.find(primitive_kind::eltwise) != -1;
        jcp.eltwise_alpha = 0.f;
    }

    auto desired_act_fmt = nhwc;
    auto desired_wei_fmt = with_groups ? (jcp.signed_input) ? gOhIw8o4i_s8s8 : gOhIw8o4i
                                       : (jcp.signed_input) ?  OhIw8o4i_s8s8 :  OhIw8o4i;

    if (src_d.format() == any)
        CHECK(src_pd.set_format(desired_act_fmt));
    if (src_d.format() != desired_act_fmt)
        return status::unimplemented;

    if (dst_d.format() == any)
        CHECK(dst_pd.set_format(desired_act_fmt));
    if (dst_d.format() != desired_act_fmt)
        return status::unimplemented;

    if (weights_d.format() == any)
        CHECK(weights_pd.set_format(desired_wei_fmt));
    if (weights_d.format() != desired_wei_fmt)
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

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    assert(IMPLICATION(!jcp.is_oc_scale, oscales.mask_ == 0));

    jcp.ur_h = 1; /* no code-unrolling by h so far */
    jcp.ur_w = isa == avx2 ? 3 : 2;
    jcp.nb_oc_blocking = 2;
    if (jcp.nb_oc % jcp.nb_oc_blocking != 0) jcp.nb_oc_blocking = 1;

    if (jcp.ow < jcp.ur_w) jcp.ur_w = jcp.ow;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    bool args_ok = true
        && jcp.l_pad <= jcp.ur_w
        && IMPLICATION(jcp.kw > 7, (jcp.t_pad == 0 && jcp.l_pad == 0)
                || (jcp.stride_w == 1 && jcp.stride_h == 1));
    if (!args_ok) return status::unimplemented;

    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
        + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));

    if (r_pad_no_tail > jcp.ur_w) {
        /* recalculate ur_w, nb_oc_blocking and ur_w_tail */
        jcp.ur_w = r_pad_no_tail + 1;
        jcp.ur_w_tail = jcp.ow % jcp.ur_w;
        /* check again ... */
        r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));
        if ((r_pad_no_tail > jcp.ur_w) || (jcp.ow < jcp.ur_w))
            return status::unimplemented;
    }
    if (jcp.l_pad > jcp.ur_w) return status::unimplemented;

    jcp.wei_adj_scale = (jcp.signed_input) ? (1.0f / 2.0f) : 1.0f;

    return status::success;
}

template struct jit_uni_x8s8s32x_conv_fwd_kernel<avx2>;
template struct jit_uni_x8s8s32x_conv_fwd_kernel<sse42>;

}
}
}
