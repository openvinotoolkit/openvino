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

#include <common/memory_tracking.hpp>
#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "cpu_memory.hpp"

#include "jit_uni_def_conv_kernel_f32.hpp"

#define GET_OFF(field) offsetof(jit_def_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;

using namespace Xbyak;

template <cpu_isa_t isa>
void jit_uni_def_conv_fwd_kernel_f32<isa>::apply_filter(int ow_step, int oc_blocks_step, int oc_step, int ic_step) {
    int repeats = isa == sse42 && oc_step > (jcp.oc_block / 2) ? 2 : 1;

    for (int kh = 0; kh < jcp.kh; kh++) {
        for (int kw = 0; kw < jcp.kw; kw++) {
            for (int ic = 0; ic < ic_step; ic++) {
                for (int ow = 0; ow < ow_step; ow++) {
                    Vmm vmm_src = get_vmm_src(ow);
                    size_t inp_off = (size_t) ow * jcp.kh * jcp.kw * jcp.ic + kh * jcp.kw * jcp.ic + kw * jcp.ic + ic;

                    uni_vbroadcastss(vmm_src, ptr[aux2_reg_input_buffer + inp_off * jcp.typesize_in]);
                }

                for (int r = 0; r < repeats; r++) {
                    for (int ocb = 0; ocb < oc_blocks_step; ocb++) {
                        Vmm vmm_ker = get_vmm_ker(0);
                        size_t ker_off = (size_t) ocb * jcp.nb_ic * jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block +
                                         kh * jcp.kw * jcp.ic_block * jcp.oc_block +
                                         kw * jcp.ic_block * jcp.oc_block +
                                         ic * jcp.oc_block + r * jcp.oc_block / 2;

                        uni_vmovups(vmm_ker, ptr[aux2_reg_kernel + ker_off * jcp.typesize_in]);
                        for (int ow = 0; ow < ow_step; ow++) {
                            Vmm vmm_src = get_vmm_src(ow);
                            Vmm vmm_acc = get_vmm_acc(r * jcp.ur_w * jcp.nb_oc_blocking + ocb * ow_step + ow);

                            if (isa == sse42 && ow > 0) {
                                uni_vmovups(vmm_ker, ptr[aux2_reg_kernel + ker_off * jcp.typesize_in]);
                            }

                            uni_vfmadd231ps(vmm_acc, vmm_ker, vmm_src);
                        }
                    }
                }
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_def_conv_fwd_kernel_f32<isa>::init_accums(int ow_step, int oc_blocks_step, int oc_step) {
    int repeats = isa == sse42 && oc_step > (jcp.oc_block / 2) ? 2 : 1;
    for (int r = 0; r < repeats; r++) {
        for (int ocb = 0; ocb < oc_blocks_step; ocb++) {
            for (int ow = 0; ow < ow_step; ow++) {
                Vmm vmm_acc = get_vmm_acc(r * jcp.ur_w * jcp.nb_oc_blocking + ocb * ow_step + ow);

                uni_vpxor(vmm_acc, vmm_acc, vmm_acc);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_def_conv_fwd_kernel_f32<isa>::ic_loop(int ow_step, int oc_blocks_step, int oc_step) {
    Label ic_main_loop;
    Label ic_tail;
    Label exit;

    push(reg_oc_work);
    push(aux_reg_bias);

    mov(aux2_reg_kernel, aux_reg_kernel);
    mov(aux2_reg_input_buffer, reg_input_buffer);

    mov(reg_ic_iter, jcp.ic);

    init_accums(ow_step, oc_blocks_step, oc_step);

    L(ic_main_loop); {
        cmp(reg_ic_iter, jcp.ic_block);
        jl(ic_tail, T_NEAR);

        apply_filter(ow_step, oc_blocks_step, oc_step, jcp.ic_block);

        add(aux2_reg_input_buffer, jcp.ic_block * jcp.typesize_in);
        add(aux2_reg_kernel, jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block * jcp.typesize_in);
        sub(reg_ic_iter, jcp.ic_block);
        jmp(ic_main_loop, T_NEAR);
    }

    L(ic_tail); {
        if (jcp.ic % jcp.ic_block != 0) {
            apply_filter(ow_step, oc_blocks_step, oc_step, jcp.ic % jcp.ic_block);
        }
    }

    pop(aux_reg_bias);
    pop(reg_oc_work);
}

template <cpu_isa_t isa>
void jit_uni_def_conv_fwd_kernel_f32<isa>::interpolate_input(int ow_step) {
    Label dg_loop;
    Label dg_loop_end;

    mov(reg_table, l_table);
    mov(aux_reg_def_off, reg_def_off);
    mov(aux_reg_input, reg_input);
    mov(aux2_reg_input_buffer, aux_reg_input_buffer);
    xor_(reg_dg_iter, reg_dg_iter);

    const int ic_per_def_group = jcp.ic / jcp.dg;
    L(dg_loop); {
        cmp(reg_dg_iter, jcp.dg);
        jge(dg_loop_end, T_NEAR);

        for (int ow = 0; ow < ow_step; ow++) {
            for (int kh = 0; kh < jcp.kh; kh++) {
                for (int kw = 0; kw < jcp.kw; kw++) {
                    Label init_with_zeros;
                    Label ic_loop_main;
                    Label ic_loop_tail;
                    Label ic_loop_zeros;
                    Label loop_end;
                    Label h_sec_opt;
                    Label h_sec_opt_exit;
                    Label w_sec_opt;
                    Label w_sec_opt_exit;

                    mov(aux2_reg_input, aux_reg_input);
                    add(aux2_reg_input, (ow * jcp.stride_w * jcp.ic) * jcp.typesize_in);

                    mov(aux3_reg_input_buffer, aux2_reg_input_buffer);
                    add(aux3_reg_input_buffer, (ow * jcp.kh * jcp.kw * jcp.ic) * jcp.typesize_in);

                    Xmm xmm_tmp = Xmm(0);

                    Xmm xmm_map_h = Xmm(2);
                    Xmm xmm_ih_in = Xmm(4);
                    Xmm xmm_ih_im = Xmm(1);
                    Xmm xmm_cur_height = xmm_ih_im;
                    Xmm xmm_h_low = xmm_ih_in;
                    Xmm xmm_h_high = xmm_cur_height;
                    Xmm xmm_lh = xmm_map_h;
                    Xmm xmm_hh = Xmm(3);

                    Xmm xmm_map_w = Xmm(6);
                    Xmm xmm_iw_in = Xmm(8);
                    Xmm xmm_iw_im = Xmm(5);
                    Xmm xmm_cur_width = xmm_iw_im;
                    Xmm xmm_w_low = xmm_iw_in;
                    Xmm xmm_w_high = xmm_cur_width;
                    Xmm xmm_lw = xmm_map_w;
                    Xmm xmm_hw = Xmm(7);

                    Xmm xmm_v1_off = Xmm(9);
                    Xmm xmm_v2_off = Xmm(10);
                    Xmm xmm_v3_off = Xmm(11);
                    Xmm xmm_v4_off = Xmm(12);

                    Xmm xmm_w1 = xmm_h_low;
                    Xmm xmm_w2 = xmm_h_high;
                    Xmm xmm_w3 = xmm_w_low;
                    Xmm xmm_w4 = xmm_w_high;

                    Xmm xmm_v1 = xmm_lh;
                    Xmm xmm_v2 = xmm_hh;
                    Xmm xmm_v3 = xmm_lw;
                    Xmm xmm_v4 = xmm_hw;

                    Vmm vmm_w1 = Vmm(xmm_h_low.getIdx());
                    Vmm vmm_w2 = Vmm(xmm_h_high.getIdx());
                    Vmm vmm_w3 = Vmm(xmm_w_low.getIdx());
                    Vmm vmm_w4 = Vmm(xmm_w_high.getIdx());

                    Vmm vmm_v1 = Vmm(xmm_lh.getIdx());
                    Vmm vmm_v2 = Vmm(xmm_hh.getIdx());
                    Vmm vmm_v3 = Vmm(xmm_lw.getIdx());
                    Vmm vmm_v4 = Vmm(xmm_hw.getIdx());

                    size_t def_off_h = ((2 * (kh * jcp.kw + kw) + 0) * jcp.oh * jcp.ow) + ow;
                    mov(reg_tmp_32, ptr[aux_reg_def_off + def_off_h * jcp.typesize_off]);
                    movq(xmm_tmp, reg_tmp_64);
                    mov(reg_tmp_32, float2int((float) (kh * (jcp.dilate_h + 1))));
                    movq(xmm_map_h, reg_tmp_64);
                    addss(xmm_map_h, xmm_tmp);

                    mov(reg_tmp_32, jcp.stride_h);
                    imul(reg_tmp_32, reg_oh_pos);
                    sub(reg_tmp_32, jcp.t_pad);
                    movq(xmm_ih_in, reg_tmp_64);

                    cvtsi2ss(xmm_ih_im, reg_tmp_32);
                    addss(xmm_ih_im, xmm_map_h);

                    movss(xmm_tmp, xmm_ih_im);
                    cmpss(xmm_tmp, table_val(0), 1);
                    movq(reg_tmp_64, xmm_tmp);
                    cmp(reg_tmp_32, 0);
                    jne(init_with_zeros, T_NEAR);

                    cmpss(xmm_ih_im, table_val(1), 1);
                    movq(reg_tmp_64, xmm_ih_im);
                    cmp(reg_tmp_32, 0);
                    je(init_with_zeros, T_NEAR);


                    size_t def_off_w = ((2 * (kh * jcp.kw + kw) + 1) * jcp.oh * jcp.ow) + ow;
                    mov(reg_tmp_32, ptr[aux_reg_def_off + def_off_w * jcp.typesize_off]);
                    movq(xmm_tmp, reg_tmp_64);
                    mov(reg_tmp_32, float2int((float) (kw * (jcp.dilate_w + 1))));
                    movq(xmm_map_w, reg_tmp_64);
                    addss(xmm_map_w, xmm_tmp);

                    mov(reg_tmp_32, jcp.stride_w);
                    imul(reg_tmp_32, reg_ow_pos);
                    sub(reg_tmp_32, jcp.l_pad - ow * jcp.stride_w);
                    movq(xmm_iw_in, reg_tmp_64);

                    cvtsi2ss(xmm_iw_im, reg_tmp_32);
                    addss(xmm_iw_im, xmm_map_w);

                    movss(xmm_tmp, xmm_iw_im);
                    cmpss(xmm_tmp, table_val(0), 1);
                    movq(reg_tmp_64, xmm_tmp);
                    cmp(reg_tmp_32, 0);
                    jne(init_with_zeros, T_NEAR);

                    cmpss(xmm_iw_im, table_val(2), 1);
                    movq(reg_tmp_64, xmm_iw_im);
                    cmp(reg_tmp_32, 0);
                    je(init_with_zeros, T_NEAR);


                    movd(xmm_cur_height, table_val(3));
                    psubd(xmm_cur_height, xmm_ih_in);

                    roundps(xmm_h_low, xmm_map_h, 1);
                    cvtps2dq(xmm_h_low, xmm_h_low);

                    movups(xmm_tmp, xmm_cur_height);
                    pcmpgtd(xmm_tmp, xmm_h_low);

                    movq(reg_tmp_64, xmm_tmp);
                    cmp(reg_tmp_32, 0);
                    jne(h_sec_opt, T_NEAR);

                    movups(xmm_h_low, xmm_cur_height);
                    movups(xmm_h_high, xmm_h_low);
                    jmp(h_sec_opt_exit);

                    L(h_sec_opt);

                    movups(xmm_h_high, xmm_h_low);
                    paddd(xmm_h_high, table_val(5));

                    L(h_sec_opt_exit);

                    cvtdq2ps(xmm_tmp, xmm_h_low);
                    subss(xmm_lh, xmm_tmp);
                    movss(xmm_hh, table_val(5));
                    cvtdq2ps(xmm_hh, xmm_hh);
                    subss(xmm_hh, xmm_lh);


                    movd(xmm_cur_width, table_val(4));
                    psubd(xmm_cur_width, xmm_iw_in);

                    roundps(xmm_w_low, xmm_map_w, 1);
                    cvtps2dq(xmm_w_low, xmm_w_low);

                    movups(xmm_tmp, xmm_cur_width);
                    pcmpgtd(xmm_tmp, xmm_w_low);

                    movq(reg_tmp_64, xmm_tmp);
                    cmp(reg_tmp_32, 0);
                    jne(w_sec_opt, T_NEAR);

                    movups(xmm_w_low, xmm_cur_width);
                    movups(xmm_w_high, xmm_w_low);
                    jmp(w_sec_opt_exit);

                    L(w_sec_opt);

                    movups(xmm_w_high, xmm_w_low);
                    paddd(xmm_w_high, table_val(5));

                    L(w_sec_opt_exit);

                    cvtdq2ps(xmm_tmp, xmm_w_low);
                    subss(xmm_lw, xmm_tmp);
                    movss(xmm_hw, table_val(5));
                    cvtdq2ps(xmm_hw, xmm_hw);
                    subss(xmm_hw, xmm_lw);


                    movups(xmm_v1_off, table_val(2));
                    cvtps2dq(xmm_v1_off, xmm_v1_off);
                    movups(xmm_v3_off, xmm_v1_off);

                    pmulld(xmm_v1_off, xmm_h_low);
                    movups(xmm_v2_off, xmm_v1_off);
                    paddd(xmm_v1_off, xmm_w_low);
                    paddd(xmm_v2_off, xmm_w_high);

                    pmulld(xmm_v3_off, xmm_h_high);
                    movups(xmm_v4_off, xmm_v3_off);
                    paddd(xmm_v3_off, xmm_w_low);
                    paddd(xmm_v4_off, xmm_w_high);


                    movss(xmm_w1, xmm_hh);
                    mulss(xmm_w1, xmm_hw);
                    uni_vbroadcastss(vmm_w1, xmm_w1);

                    movss(xmm_w2, xmm_hh);
                    mulss(xmm_w2, xmm_lw);
                    uni_vbroadcastss(vmm_w2, xmm_w2);

                    movss(xmm_w3, xmm_lh);
                    mulss(xmm_w3, xmm_hw);
                    uni_vbroadcastss(vmm_w3, xmm_w3);

                    movss(xmm_w4, xmm_lh);
                    mulss(xmm_w4, xmm_lw);
                    uni_vbroadcastss(vmm_w4, xmm_w4);

                    int simd_w = vlen / jcp.typesize_in;
                    mov(reg_ic_iter, ic_per_def_group);
                    L(ic_loop_main);
                    {
                        cmp(reg_ic_iter, simd_w);
                        jl(ic_loop_tail, T_NEAR);

                        size_t input_buffer_off = (size_t) kh * jcp.kw * jcp.ic + kw * jcp.ic;

                        pmovsxdq(xmm_v1_off, xmm_v1_off);
                        movq(reg_tmp_64, xmm_v1_off);
                        imul(reg_tmp_64, reg_tmp_64, jcp.ic * jcp.typesize_in);
                        add(reg_tmp_64, aux2_reg_input);
                        uni_vmovups(vmm_v1, ptr[reg_tmp_64]);
                        uni_vmulps(vmm_v1, vmm_v1, vmm_w1);

                        pmovsxdq(xmm_v2_off, xmm_v2_off);
                        movq(reg_tmp_64, xmm_v2_off);
                        imul(reg_tmp_64, reg_tmp_64, jcp.ic * jcp.typesize_in);
                        add(reg_tmp_64, aux2_reg_input);
                        uni_vmovups(vmm_v2, ptr[reg_tmp_64]);
                        uni_vmulps(vmm_v2, vmm_v2, vmm_w2);

                        pmovsxdq(xmm_v3_off, xmm_v3_off);
                        movq(reg_tmp_64, xmm_v3_off);
                        imul(reg_tmp_64, reg_tmp_64, jcp.ic * jcp.typesize_in);
                        add(reg_tmp_64, aux2_reg_input);
                        uni_vmovups(vmm_v3, ptr[reg_tmp_64]);
                        uni_vmulps(vmm_v3, vmm_v3, vmm_w3);

                        pmovsxdq(xmm_v4_off, xmm_v4_off);
                        movq(reg_tmp_64, xmm_v4_off);
                        imul(reg_tmp_64, reg_tmp_64, jcp.ic * jcp.typesize_in);
                        add(reg_tmp_64, aux2_reg_input);
                        uni_vmovups(vmm_v4, ptr[reg_tmp_64]);
                        uni_vmulps(vmm_v4, vmm_v4, vmm_w4);

                        uni_vaddps(vmm_v1, vmm_v1, vmm_v2);
                        uni_vaddps(vmm_v1, vmm_v1, vmm_v3);
                        uni_vaddps(vmm_v1, vmm_v1, vmm_v4);
                        uni_vmovups(ptr[aux3_reg_input_buffer + input_buffer_off * jcp.typesize_in], vmm_v1);

                        add(aux2_reg_input, simd_w * jcp.typesize_in);
                        add(aux3_reg_input_buffer, simd_w * jcp.typesize_in);
                        sub(reg_ic_iter, simd_w);
                        jmp(ic_loop_main, T_NEAR);
                    };

                    L(ic_loop_tail);
                    {
                        cmp(reg_ic_iter, 1);
                        jl(loop_end, T_NEAR);

                        size_t input_buffer_off = (size_t) kh * jcp.kw * jcp.ic + kw * jcp.ic;

                        pmovsxdq(xmm_v1_off, xmm_v1_off);
                        movq(reg_tmp_64, xmm_v1_off);
                        imul(reg_tmp_64, reg_tmp_64, jcp.ic * jcp.typesize_in);
                        add(reg_tmp_64, aux2_reg_input);
                        movss(xmm_v1, ptr[reg_tmp_64]);
                        mulss(xmm_v1, xmm_w1);

                        pmovsxdq(xmm_v2_off, xmm_v2_off);
                        movq(reg_tmp_64, xmm_v2_off);
                        imul(reg_tmp_64, reg_tmp_64, jcp.ic * jcp.typesize_in);
                        add(reg_tmp_64, aux2_reg_input);
                        movss(xmm_v2, ptr[reg_tmp_64]);
                        mulss(xmm_v2, xmm_w2);

                        pmovsxdq(xmm_v3_off, xmm_v3_off);
                        movq(reg_tmp_64, xmm_v3_off);
                        imul(reg_tmp_64, reg_tmp_64, jcp.ic * jcp.typesize_in);
                        add(reg_tmp_64, aux2_reg_input);
                        movss(xmm_v3, ptr[reg_tmp_64]);
                        mulss(xmm_v3, xmm_w3);

                        pmovsxdq(xmm_v4_off, xmm_v4_off);
                        movq(reg_tmp_64, xmm_v4_off);
                        imul(reg_tmp_64, reg_tmp_64, jcp.ic * jcp.typesize_in);
                        add(reg_tmp_64, aux2_reg_input);
                        movss(xmm_v4, ptr[reg_tmp_64]);
                        mulss(xmm_v4, xmm_w4);

                        addss(xmm_v1, xmm_v2);
                        addss(xmm_v1, xmm_v3);
                        addss(xmm_v1, xmm_v4);
                        movss(ptr[aux3_reg_input_buffer + input_buffer_off * jcp.typesize_in], xmm_v1);

                        add(aux2_reg_input, jcp.typesize_in);
                        add(aux3_reg_input_buffer, jcp.typesize_in);
                        sub(reg_ic_iter, 1);
                        jmp(ic_loop_tail, T_NEAR);
                    };

                    jmp(loop_end, T_NEAR);

                    L(init_with_zeros);

                    mov(reg_ic_iter, 0);
                    L(ic_loop_zeros);
                    {
                        cmp(reg_ic_iter, ic_per_def_group);
                        je(loop_end, T_NEAR);

                        size_t input_buffer_off = (size_t) kh * jcp.kw * jcp.ic + kw * jcp.ic;

                        pxor(xmm_tmp, xmm_tmp);
                        movss(ptr[aux3_reg_input_buffer + input_buffer_off * jcp.typesize_in], xmm_tmp);
                        add(aux3_reg_input_buffer, jcp.typesize_in);
                        inc(reg_ic_iter);
                        jmp(ic_loop_zeros, T_NEAR);
                    }

                    L(loop_end);
                }
            }
        }

        add(aux_reg_def_off, 2 * jcp.kh * jcp.kw * jcp.oh * jcp.ow * jcp.typesize_off);
        add(aux_reg_input, ic_per_def_group * jcp.typesize_in);
        add(aux2_reg_input_buffer, ic_per_def_group * jcp.typesize_in);
        inc(reg_dg_iter);
        jmp(dg_loop, T_NEAR);
    }

    L(dg_loop_end);
}

template <cpu_isa_t isa>
void jit_uni_def_conv_fwd_kernel_f32<isa>::store_output(int ow_step, int oc_blocks_step, int oc_step) {
    int repeats = isa == sse42 && oc_step > (jcp.oc_block / 2) ? 2 : 1;

    if (jcp.with_bias) {
        for (int r = 0; r < repeats; r++) {
            for (int ocb = 0; ocb < oc_blocks_step; ocb++) {
                size_t bias_off = (size_t) ocb * jcp.oc_block + r * jcp.oc_block / 2;
                uni_vmovups(Vmm(0), ptr[aux_reg_bias + bias_off * jcp.typesize_bia]);

                for (int ow = 0; ow < ow_step; ow++) {
                    Vmm vmm_acc = get_vmm_acc(r * jcp.ur_w * jcp.nb_oc_blocking + ocb * ow_step + ow);

                    uni_vaddps(vmm_acc, vmm_acc, Vmm(0));
                }
            }
        }
    }

    if (isa == avx512_common && oc_step != jcp.oc_block) {
        int mask = (1 << oc_step) - 1;
        mov(reg_tmp_32, mask);
        kmovw(ktail_mask, reg_tmp_32);
    }

    for (int r = 0; r < repeats; r++) {
        int tail_size = isa == sse42 ? nstl::min(jcp.oc_block / 2, oc_step - r * jcp.oc_block / 2) : oc_step;
        bool is_scalar_store = isa == sse42 ? tail_size < jcp.oc_block / 2 : tail_size < jcp.oc_block;
        if (is_scalar_store) {
            for (int ow = 0; ow < ow_step; ow++) {
                Vmm vmm_dst = get_vmm_acc(r * jcp.ur_w * jcp.nb_oc_blocking + ow);
                Xmm xmm_dst = get_xmm_acc(r * jcp.ur_w * jcp.nb_oc_blocking + ow);

                if (isa == avx512_common) {
                    size_t out_off = (size_t) ow * jcp.oc;

                    uni_vmovups(ptr[aux_reg_output + out_off * jcp.typesize_out], vmm_dst | ktail_mask);
                } else {
                    for (int oc = 0; oc < tail_size; oc++) {
                        size_t out_off = (size_t) ow * jcp.oc + oc + r * (jcp.oc_block / 2);

                        movq(reg_tmp_64, xmm_dst);
                        mov(ptr[aux_reg_output + out_off * jcp.typesize_out], reg_tmp_32);

                        if (isa == sse42) {
                            psrldq(vmm_dst, jcp.typesize_out);
                        } else {
                            Ymm ymm_dst = get_ymm_acc(ow);
                            Vmm vmm_tmp = Vmm(0);
                            Ymm ymm_tmp = Ymm(0);

                            vperm2i128(ymm_tmp, ymm_dst, ymm_dst, 0x01);
                            vpalignr(ymm_dst, vmm_tmp, ymm_dst, jcp.typesize_out);
                        }
                    }
                }
            }
        } else {
            for (int ocb = 0; ocb < oc_blocks_step; ocb++) {
                for (int ow = 0; ow < ow_step; ow++) {
                    Vmm vmm_acc = get_vmm_acc(r * jcp.ur_w * jcp.nb_oc_blocking + ocb * ow_step + ow);
                    size_t out_off = (size_t) ow * jcp.oc + ocb * jcp.oc_block + r * (jcp.oc_block / 2);

                    uni_vmovups(ptr[aux_reg_output + out_off * jcp.typesize_out], vmm_acc);
                }
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_def_conv_fwd_kernel_f32<isa>::oc_loop(int ow_step) {
    Label oc_unrolled_loop;
    Label oc_main_loop;
    Label oc_tail;

    mov(aux_reg_input_buffer, reg_input_buffer);

    push(reg_output);
    push(reg_bias);
    push(reg_input);
    push(reg_kernel);

    interpolate_input(ow_step);

    pop(reg_kernel);
    pop(reg_input);
    pop(reg_bias);
    pop(reg_output);

    push(reg_ow_pos);

    mov(aux_reg_kernel, reg_kernel);
    mov(aux_reg_output, reg_output);
    mov(aux_reg_bias, reg_bias);

    mov(reg_oc_work, jcp.oc);

    L(oc_unrolled_loop); {
        cmp(reg_oc_work, jcp.nb_oc_blocking * jcp.oc_block);
        jl(oc_main_loop, T_NEAR);

        ic_loop(ow_step, jcp.nb_oc_blocking, jcp.oc_block);
        store_output(ow_step, jcp.nb_oc_blocking, jcp.oc_block);

        add(aux_reg_kernel, jcp.nb_oc_blocking * jcp.nb_ic * jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block * jcp.typesize_in);
        add(aux_reg_output, jcp.nb_oc_blocking * jcp.oc_block * jcp.typesize_out);
        add(aux_reg_bias, jcp.nb_oc_blocking * jcp.oc_block * jcp.typesize_bia);
        sub(reg_oc_work, jcp.nb_oc_blocking * jcp.oc_block);

        jmp(oc_unrolled_loop, T_NEAR);
    }

    L(oc_main_loop); {
        cmp(reg_oc_work, jcp.oc_block);
        jl(oc_tail, T_NEAR);

        ic_loop(ow_step, 1, jcp.oc_block);
        store_output(ow_step, 1, jcp.oc_block);

        add(aux_reg_kernel, jcp.nb_ic * jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block * jcp.typesize_in);
        add(aux_reg_output, jcp.oc_block * jcp.typesize_out);
        add(aux_reg_bias, jcp.oc_block * jcp.typesize_bia);
        sub(reg_oc_work, jcp.oc_block);

        jmp(oc_main_loop, T_NEAR);
    }

    L(oc_tail); {
        if (jcp.oc % jcp.oc_block != 0) {
            ic_loop(ow_step, 1, jcp.oc % jcp.oc_block);
            store_output(ow_step, 1, jcp.oc % jcp.oc_block);
        }
    }

    pop(reg_ow_pos);
}

template <cpu_isa_t isa>
void jit_uni_def_conv_fwd_kernel_f32<isa>::ow_loop() {
    Label ow_loop_main;
    Label ow_tail;

    mov(reg_ow_pos, 0);

    L(ow_loop_main); {
        cmp(reg_ow_pos, jcp.ow - jcp.ur_w);
        jg(ow_tail, T_NEAR);

        oc_loop(jcp.ur_w);

        add(reg_input, jcp.ur_w * jcp.stride_w * jcp.ic * jcp.typesize_in);
        add(reg_def_off, jcp.ur_w * jcp.typesize_off);
        add(reg_output, jcp.ur_w * jcp.oc * jcp.typesize_out);

        add(reg_ow_pos, jcp.ur_w);
        jmp(ow_loop_main, T_NEAR);
    }

    L(ow_tail); {
        if (jcp.ow % jcp.ur_w != 0)
            oc_loop(jcp.ow % jcp.ur_w);
    }
}

template <cpu_isa_t isa>
void jit_uni_def_conv_fwd_kernel_f32<isa>::generate()
{
    this->preamble();

    mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_def_off, ptr[this->param1 + GET_OFF(off)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    if (jcp.with_bias)
        mov(reg_bias, ptr[this->param1 + GET_OFF(bias)]);
    mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_input_buffer, ptr[this->param1 + GET_OFF(buf)]);
    mov(reg_oh_pos, ptr[param1 + GET_OFF(oh_pos)]);

    ow_loop();

    this->postamble();

    prepare_table();
}

template <cpu_isa_t isa>
void jit_uni_def_conv_fwd_kernel_f32<isa>::prepare_table() {
    align(64);
    L(l_table);
    for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
        dd(0);
    }

    for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
        dd(float2int((float)jcp.ih));
    }

    for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
        dd(float2int((float)jcp.iw));
    }

    for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
        dd(jcp.ih - 1);
    }

    for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
        dd(jcp.iw - 1);
    }

    for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
        dd(1);
    }
}

template <cpu_isa_t isa>
bool jit_uni_def_conv_fwd_kernel_f32<isa>::post_ops_ok(jit_def_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    return p.len_ == 0;
}

template <cpu_isa_t isa>
status_t jit_uni_def_conv_fwd_kernel_f32<isa>::init_conf(jit_def_conv_conf_t &jcp,
        const deformable_convolution_desc_t &cd, cpu_memory_t::pd_t &src_pd,
        cpu_memory_t::pd_t &offsets_pd, cpu_memory_t::pd_t &weights_pd,
        cpu_memory_t::pd_t &dst_pd, cpu_memory_t::pd_t &bias_pd,
        const primitive_attr_t &attr)
{
    if (!mayiuse(isa)) return status::unimplemented;

    const memory_desc_wrapper src_d(&src_pd);
    const memory_desc_wrapper offsets_d(&offsets_pd);
    const memory_desc_wrapper weights_d(&weights_pd);
    const memory_desc_wrapper dst_d(&dst_pd);
    const memory_desc_wrapper bias_d(&bias_pd);

    jcp.prop_kind = cd.prop_kind;

    jcp.dg = cd.deformable_group;

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

    jcp.with_bias = cd.bias_desc.format != memory_format::undef;

    const int simd_w = isa == avx512_common ? 16 : 8;
    jcp.ic_block = simd_w;
    jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);

    jcp.oc_block = simd_w;
    jcp.oc_padded = rnd_up(jcp.oc, jcp.oc_block);
    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);

    if (jcp.ngroups != 1)
        return status::unimplemented;

    if (jcp.ic % jcp.dg != 0)
        return status::unimplemented;

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    auto desired_act_fmt = nhwc;
    auto desired_off_fmt = nchw;
    auto desired_wei_fmt = with_groups ? isa == avx512_common ? gOIhw16i16o : gOIhw8i8o
                                       : isa == avx512_common ? OIhw16i16o : OIhw8i8o;

    if (src_d.format() == any)
        CHECK(src_pd.set_format(desired_act_fmt));
    if (src_d.format() != desired_act_fmt)
        return status::unimplemented;

    if (offsets_d.format() == any)
        CHECK(offsets_pd.set_format(desired_off_fmt));
    if (offsets_d.format() != desired_off_fmt)
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

    if (dst_d.format() == any)
        CHECK(dst_pd.set_format(desired_act_fmt));
    if (dst_d.format() != desired_act_fmt)
        return status::unimplemented;

    jcp.src_dt = cd.src_descs[0].data_type;
    jcp.off_dt = cd.src_descs[1].data_type;
    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;

    jcp.typesize_in = (int)types::data_type_size(jcp.src_dt);
    jcp.typesize_off = (int)types::data_type_size(jcp.off_dt);
    jcp.typesize_out = (int)types::data_type_size(jcp.dst_dt);
    jcp.typesize_bia = (int)(jcp.with_bias ? types::data_type_size(jcp.bia_dt) : 0);

    jcp.ur_w = isa == avx512_common ? 6 : 3;
    jcp.nb_oc_blocking = isa == sse42 ? 2 : 4;

    jcp.nthr = mkldnn_get_max_threads();

    return status::success;
}

template <cpu_isa_t isa>
void jit_uni_def_conv_fwd_kernel_f32<isa>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_def_conv_conf_t &jcp, const primitive_attr_t &attr) {

    scratchpad.book(key_def_conv_buffer, (size_t)jcp.nthr * jcp.ur_w * jcp.kh * jcp.kw * jcp.ic * jcp.typesize_in);
    if (jcp.oc != jcp.oc_padded) {
        scratchpad.book(key_conv_padded_bias, (size_t)jcp.typesize_bia * jcp.oc_padded);
    }
}

template struct jit_uni_def_conv_fwd_kernel_f32<avx512_common>;
template struct jit_uni_def_conv_fwd_kernel_f32<avx2>;
template struct jit_uni_def_conv_fwd_kernel_f32<sse42>;

}
}
}
