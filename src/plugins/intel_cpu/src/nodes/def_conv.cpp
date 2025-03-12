// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "def_conv.h"

#include <cmath>
#include <common/dnnl_thread.hpp>
#include <memory>
#include <openvino/op/deformable_convolution.hpp>
#include <string>
#include <vector>

#include "common/primitive_hashing_utils.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "dnnl_extension_utils.h"
#include "dnnl_types.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "openvino/core/parallel.hpp"
#include "openvino/util/pp.hpp"

using namespace dnnl;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace ov::intel_cpu::node {
#if defined(OPENVINO_ARCH_X86_64)
#    define GET_OFF(field) offsetof(jit_def_conv_call_args, field)

template <cpu_isa_t isa>
struct jit_uni_def_conv_kernel_f32 : public jit_uni_def_conv_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_def_conv_kernel_f32)

    constexpr static int sampledPointsPerPixel = DeformableConvolution::sampledPointsPerPixel;

    explicit jit_uni_def_conv_kernel_f32(const jit_def_conv_params& jcp)
        : jit_uni_def_conv_kernel(jcp),
          jit_generator(jit_name()) {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    };

    void generate() override {
        this->preamble();

        mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
        mov(reg_sampled_wei, ptr[this->param1 + GET_OFF(sampledWei)]);
        mov(reg_sampled_offs, ptr[this->param1 + GET_OFF(sampledCoords)]);

        mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
        if (jcp_.with_bias) {
            mov(reg_bias, ptr[this->param1 + GET_OFF(bias)]);
        }
        mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
        mov(reg_input_buffer_temp, ptr[this->param1 + GET_OFF(buf)]);
        mov(oh_pos_temp, ptr[param1 + GET_OFF(oh_pos)]);

        // need to save temporary to prevent using of %rdi during GET_OFF(...)
        mov(reg_oh_pos, oh_pos_temp);
        // prevents mismatching param1 == %rcx (on windows) and reg_input_buffer
        mov(reg_input_buffer, reg_input_buffer_temp);

        ow_loop();

        this->postamble();

        prepare_table();
    }

private:
    using Vmm =
        typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;

    const int vlen = cpu_isa_traits<isa>::vlen;
    using Ymm = const Xbyak::Ymm;
    using Xmm = const Xbyak::Xmm;
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using reg8_t = const Xbyak::Reg8;

    reg64_t reg_input = r8;
    reg64_t reg_sampled_wei = r9;
    reg64_t reg_kernel = r10;
    reg64_t reg_bias = r11;
    reg64_t reg_output = r12;
    reg64_t reg_oh_pos = rdi;
    reg64_t aux_reg_bias = rsi;
    reg64_t reg_ow_pos = rdx;
    reg64_t aux_reg_output = reg_ow_pos;
    reg64_t reg_dg_iter = reg_output;
    reg64_t aux_reg_input = rax;
    reg64_t aux2_reg_input = reg_kernel;
    reg64_t reg_ic_iter = rbx;
    reg64_t reg_oc_work = reg_ic_iter;
    reg64_t aux_reg_sampled_wei = reg_bias;
    reg64_t reg_input_buffer = rcx;
    reg64_t aux_reg_input_buffer = r14;
    reg32_t reg_tmp_32 = r15d;
    reg64_t reg_tmp_64 = r15;
    reg64_t reg_table = rbp;
    reg64_t aux_reg_kernel = reg_table;
    reg64_t aux2_reg_kernel = r15;
    reg64_t oh_pos_temp = aux2_reg_kernel;
    reg64_t aux2_reg_input_buffer = aux_reg_bias;
    reg64_t reg_sampled_offs = aux2_reg_input_buffer;
    reg64_t aux3_reg_input_buffer = reg_input;
    reg64_t aux_reg_sampled_offs = r13;
    reg64_t reg_input_buffer_temp = aux_reg_sampled_offs;

    Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);

    inline Xbyak::Address table_val(int index) {
        return ptr[reg_table + index * vlen];
    }

    inline Vmm get_vmm_ker(int idx) {
        return Vmm(idx + 0);
    }
    inline Vmm get_vmm_src(int idx) {
        return Vmm(idx + 1);
    }
    inline Vmm get_vmm_acc(int idx) {
        return Vmm(idx + jcp_.ur_w + 1);
    }
    inline Ymm get_ymm_acc(int idx) {
        return Ymm(idx + jcp_.ur_w + 1);
    }
    inline Xmm get_xmm_acc(int idx) {
        return Xmm(idx + jcp_.ur_w + 1);
    }

    Xbyak::Label l_table;

    inline void checkZeroWei(const Xbyak::Xmm& x1, Label& nullifyLabel) {
        ptest(x1, x1);
        jz(nullifyLabel);
    }

    void ow_loop() {
        Label ow_loop_main;
        Label ow_tail;

        mov(reg_ow_pos, 0);

        L(ow_loop_main);
        {
            cmp(reg_ow_pos, jcp_.ow - jcp_.ur_w);
            jg(ow_tail, T_NEAR);

            oc_loop(jcp_.ur_w);
            add(reg_sampled_wei,
                jcp_.ur_w * jcp_.kh * jcp_.kw * sampledPointsPerPixel * jcp_.typesize_sampled_wei);  // type = float
            add(reg_sampled_offs,
                jcp_.ur_w * jcp_.kh * jcp_.kw * sampledPointsPerPixel * jcp_.typesize_sampled_offsets);  // type = int

            add(reg_output, jcp_.ur_w * jcp_.oc * jcp_.typesize_out);

            add(reg_ow_pos, jcp_.ur_w);
            jmp(ow_loop_main, T_NEAR);
        }

        L(ow_tail);
        {
            if (jcp_.ow % jcp_.ur_w != 0) {
                oc_loop(jcp_.ow % jcp_.ur_w);
            }
        }
    }

    void prepare_table() {
        align(64);
        L(l_table);
        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(0);
        }

        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(cpu::x64::float2int(static_cast<float>(jcp_.ih)));
        }

        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(cpu::x64::float2int(static_cast<float>(jcp_.iw)));
        }

        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(jcp_.ih - 1);
        }

        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(jcp_.iw - 1);
        }

        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(1);
        }
    }

    void apply_filter(int ow_step, int oc_blocks_step, int oc_step, int ic_step) {
        int repeats = isa == cpu::x64::sse41 && oc_step > (jcp_.oc_block / 2) ? 2 : 1;

        for (int kh = 0; kh < jcp_.kh; kh++) {
            for (int kw = 0; kw < jcp_.kw; kw++) {
                for (int ic = 0; ic < ic_step; ic++) {
                    for (int ow = 0; ow < ow_step; ow++) {
                        Vmm vmm_src = get_vmm_src(ow);
                        size_t inp_off = static_cast<size_t>(ow) * jcp_.kh * jcp_.kw * jcp_.ic +
                                         kh * jcp_.kw * jcp_.ic + kw * jcp_.ic + ic;

                        uni_vbroadcastss(vmm_src, ptr[aux2_reg_input_buffer + inp_off * jcp_.typesize_in]);
                    }

                    for (int r = 0; r < repeats; r++) {
                        for (int ocb = 0; ocb < oc_blocks_step; ocb++) {
                            Vmm vmm_ker = get_vmm_ker(0);
                            size_t ker_off = static_cast<size_t>(ocb) * jcp_.nb_ic * jcp_.kh * jcp_.kw * jcp_.ic_block *
                                                 jcp_.oc_block +
                                             kh * jcp_.kw * jcp_.ic_block * jcp_.oc_block +
                                             kw * jcp_.ic_block * jcp_.oc_block + ic * jcp_.oc_block +
                                             r * jcp_.oc_block / 2;

                            uni_vmovups(vmm_ker, ptr[aux2_reg_kernel + ker_off * jcp_.typesize_in]);
                            for (int ow = 0; ow < ow_step; ow++) {
                                Vmm vmm_src = get_vmm_src(ow);
                                Vmm vmm_acc = get_vmm_acc(r * jcp_.ur_w * jcp_.nb_oc_blocking + ocb * ow_step + ow);

                                if (isa == cpu::x64::sse41 && ow > 0) {
                                    uni_vmovups(vmm_ker, ptr[aux2_reg_kernel + ker_off * jcp_.typesize_in]);
                                }
                                uni_vfmadd231ps(vmm_acc, vmm_ker, vmm_src);
                            }
                        }
                    }
                }
            }
        }
    }

    void init_accums(int ow_step, int oc_blocks_step, int oc_step) {
        int repeats = isa == cpu::x64::sse41 && oc_step > (jcp_.oc_block / 2) ? 2 : 1;
        for (int r = 0; r < repeats; r++) {
            for (int ocb = 0; ocb < oc_blocks_step; ocb++) {
                for (int ow = 0; ow < ow_step; ow++) {
                    Vmm vmm_acc = get_vmm_acc(r * jcp_.ur_w * jcp_.nb_oc_blocking + ocb * ow_step + ow);
                    uni_vpxor(vmm_acc, vmm_acc, vmm_acc);
                }
            }
        }
    }

    void ic_loop(int ow_step, int oc_blocks_step, int oc_step) {
        Label ic_main_loop;
        Label ic_tail;
        Label exit;

        push(reg_oc_work);
        push(aux_reg_bias);
        push(reg_sampled_offs);

        mov(aux2_reg_kernel, aux_reg_kernel);
        mov(aux2_reg_input_buffer, reg_input_buffer);
        mov(reg_ic_iter, jcp_.ic);

        init_accums(ow_step, oc_blocks_step, oc_step);

        L(ic_main_loop);
        {
            cmp(reg_ic_iter, jcp_.ic_block);
            jl(ic_tail, T_NEAR);

            apply_filter(ow_step, oc_blocks_step, oc_step, jcp_.ic_block);
            add(aux2_reg_input_buffer, jcp_.ic_block * jcp_.typesize_in);
            add(aux2_reg_kernel, jcp_.kh * jcp_.kw * jcp_.ic_block * jcp_.oc_block * jcp_.typesize_in);
            sub(reg_ic_iter, jcp_.ic_block);
            jmp(ic_main_loop, T_NEAR);
        }

        L(ic_tail);
        {
            if (jcp_.ic % jcp_.ic_block != 0) {
                apply_filter(ow_step, oc_blocks_step, oc_step, jcp_.ic % jcp_.ic_block);
            }
        }

        pop(reg_sampled_offs);
        pop(aux_reg_bias);
        pop(reg_oc_work);
    }

    void prepare_buffer(int ow_step) {
        Label dg_loop;
        Label dg_loop_end;

        mov(reg_table, l_table);
        mov(aux_reg_sampled_wei, reg_sampled_wei);
        mov(aux_reg_sampled_offs, reg_sampled_offs);
        mov(aux_reg_input, reg_input);
        push(reg_sampled_offs);
        mov(aux2_reg_input_buffer, aux_reg_input_buffer);
        xor_(reg_dg_iter, reg_dg_iter);

        const int ic_per_def_group = jcp_.ic / jcp_.dg;
        L(dg_loop);
        {
            cmp(reg_dg_iter, jcp_.dg);
            jge(dg_loop_end, T_NEAR);

            for (int ow = 0; ow < ow_step; ow++) {
                for (int kh = 0; kh < jcp_.kh; kh++) {
                    for (int kw = 0; kw < jcp_.kw; kw++) {
                        Label ic_loop_main;
                        Label ic_loop_tail;
                        Label loop_end;
                        Label nullify_v1;
                        Label nullify_v2;
                        Label nullify_v3;
                        Label nullify_v4;
                        Label nullify_v1_end;
                        Label nullify_v2_end;
                        Label nullify_v3_end;
                        Label nullify_v4_end;
                        Label nullify_v1_tail;
                        Label nullify_v2_tail;
                        Label nullify_v3_tail;
                        Label nullify_v4_tail;
                        Label nullify_v1_end_tail;
                        Label nullify_v2_end_tail;
                        Label nullify_v3_end_tail;
                        Label nullify_v4_end_tail;

                        mov(aux2_reg_input, aux_reg_input);

                        mov(aux3_reg_input_buffer, aux2_reg_input_buffer);
                        add(aux3_reg_input_buffer, (ow * jcp_.kh * jcp_.kw * jcp_.ic) * jcp_.typesize_in);

                        auto xmm_v1_off = Xmm(9);
                        auto xmm_v2_off = Xmm(10);
                        auto xmm_v3_off = Xmm(11);
                        auto xmm_v4_off = Xmm(12);

                        auto xmm_w1 = Xmm(4);
                        auto xmm_w2 = Xmm(1);
                        auto xmm_w3 = Xmm(8);
                        auto xmm_w4 = Xmm(5);

                        auto xmm_v1 = Xmm(2);
                        auto xmm_v2 = Xmm(3);
                        ;
                        auto xmm_v3 = Xmm(6);
                        auto xmm_v4 = Xmm(7);

                        auto vmm_w1 = Vmm(xmm_w1.getIdx());
                        auto vmm_w2 = Vmm(xmm_w2.getIdx());
                        auto vmm_w3 = Vmm(xmm_w3.getIdx());
                        auto vmm_w4 = Vmm(xmm_w4.getIdx());

                        auto vmm_v1 = Vmm(xmm_v1.getIdx());
                        auto vmm_v2 = Vmm(xmm_v2.getIdx());
                        auto vmm_v3 = Vmm(xmm_v3.getIdx());
                        auto vmm_v4 = Vmm(xmm_v4.getIdx());

                        // offsets computation
                        size_t ind_off_hh = sampledPointsPerPixel *
                                            ((static_cast<size_t>(kh) * jcp_.kw + kw) + ow * (jcp_.kh * jcp_.kw));
                        size_t ind_off_hl = ind_off_hh + 1;
                        size_t ind_off_lh = ind_off_hl + 1;
                        size_t ind_off_ll = ind_off_lh + 1;

                        uni_vmovd(xmm_v1_off, dword[aux_reg_sampled_offs + ind_off_ll * jcp_.typesize_sampled_offsets]);
                        uni_vmovd(xmm_v2_off, dword[aux_reg_sampled_offs + ind_off_hl * jcp_.typesize_sampled_offsets]);
                        uni_vmovd(xmm_v3_off, dword[aux_reg_sampled_offs + ind_off_lh * jcp_.typesize_sampled_offsets]);
                        uni_vmovd(xmm_v4_off, dword[aux_reg_sampled_offs + ind_off_hh * jcp_.typesize_sampled_offsets]);

                        // w's computation
                        uni_vbroadcastss(vmm_w1, dword[aux_reg_sampled_wei + ind_off_ll * jcp_.typesize_sampled_wei]);
                        uni_vbroadcastss(vmm_w2, dword[aux_reg_sampled_wei + ind_off_hl * jcp_.typesize_sampled_wei]);
                        uni_vbroadcastss(vmm_w3, dword[aux_reg_sampled_wei + ind_off_lh * jcp_.typesize_sampled_wei]);
                        uni_vbroadcastss(vmm_w4, dword[aux_reg_sampled_wei + ind_off_hh * jcp_.typesize_sampled_wei]);

                        int simd_w = vlen / jcp_.typesize_in;
                        mov(reg_ic_iter, ic_per_def_group);

                        L(ic_loop_main);
                        {
                            cmp(reg_ic_iter, simd_w);
                            jl(ic_loop_tail, T_NEAR);

                            // check zero markers
                            uni_vbroadcastss(xmm_v1,
                                             dword[aux_reg_sampled_wei + ind_off_ll * jcp_.typesize_sampled_wei]);
                            uni_vbroadcastss(xmm_v2,
                                             dword[aux_reg_sampled_wei + ind_off_hl * jcp_.typesize_sampled_wei]);
                            uni_vbroadcastss(xmm_v3,
                                             dword[aux_reg_sampled_wei + ind_off_lh * jcp_.typesize_sampled_wei]);
                            uni_vbroadcastss(xmm_v4,
                                             dword[aux_reg_sampled_wei + ind_off_hh * jcp_.typesize_sampled_wei]);

                            size_t input_buffer_off = static_cast<size_t>(kh) * jcp_.kw * jcp_.ic + kw * jcp_.ic;

                            uni_vpmovsxdq(xmm_v1_off, xmm_v1_off);
                            uni_vmovq(reg_tmp_64, xmm_v1_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);
                            checkZeroWei(xmm_v1, nullify_v1);
                            uni_vmovups(vmm_v1, ptr[reg_tmp_64]);
                            uni_vmulps(vmm_v1, vmm_v1, vmm_w1);
                            jmp(nullify_v1_end, T_NEAR);
                            L(nullify_v1);
                            { uni_vpxor(vmm_v1, vmm_v1, vmm_v1); }
                            L(nullify_v1_end);

                            uni_vpmovsxdq(xmm_v2_off, xmm_v2_off);
                            uni_vmovq(reg_tmp_64, xmm_v2_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);
                            checkZeroWei(xmm_v2, nullify_v2);
                            uni_vmovups(vmm_v2, ptr[reg_tmp_64]);
                            uni_vmulps(vmm_v2, vmm_v2, vmm_w2);
                            jmp(nullify_v2_end, T_NEAR);
                            L(nullify_v2);
                            { uni_vpxor(vmm_v2, vmm_v2, vmm_v2); }
                            L(nullify_v2_end);

                            uni_vpmovsxdq(xmm_v3_off, xmm_v3_off);
                            uni_vmovq(reg_tmp_64, xmm_v3_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);
                            checkZeroWei(xmm_v3, nullify_v3);
                            uni_vmovups(vmm_v3, ptr[reg_tmp_64]);
                            uni_vmulps(vmm_v3, vmm_v3, vmm_w3);
                            jmp(nullify_v3_end, T_NEAR);
                            L(nullify_v3);
                            { uni_vpxor(vmm_v3, vmm_v3, vmm_v3); }
                            L(nullify_v3_end);

                            uni_vpmovsxdq(xmm_v4_off, xmm_v4_off);
                            uni_vmovq(reg_tmp_64, xmm_v4_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);
                            checkZeroWei(xmm_v4, nullify_v4);
                            uni_vmovups(vmm_v4, ptr[reg_tmp_64]);
                            uni_vmulps(vmm_v4, vmm_v4, vmm_w4);
                            jmp(nullify_v4_end, T_NEAR);
                            L(nullify_v4);
                            { uni_vpxor(vmm_v4, vmm_v4, vmm_v4); }
                            L(nullify_v4_end);

                            uni_vaddps(vmm_v1, vmm_v1, vmm_v2);
                            uni_vaddps(vmm_v1, vmm_v1, vmm_v3);
                            uni_vaddps(vmm_v1, vmm_v1, vmm_v4);
                            uni_vmovups(ptr[aux3_reg_input_buffer + input_buffer_off * jcp_.typesize_in], vmm_v1);

                            add(aux2_reg_input, simd_w * jcp_.typesize_in);
                            add(aux3_reg_input_buffer, simd_w * jcp_.typesize_in);
                            sub(reg_ic_iter, simd_w);
                            jmp(ic_loop_main, T_NEAR);
                        }

                        L(ic_loop_tail);
                        {
                            cmp(reg_ic_iter, 1);
                            jl(loop_end, T_NEAR);

                            // check zero markers
                            uni_vbroadcastss(xmm_v1,
                                             dword[aux_reg_sampled_wei + ind_off_ll * jcp_.typesize_sampled_wei]);
                            uni_vbroadcastss(xmm_v2,
                                             dword[aux_reg_sampled_wei + ind_off_hl * jcp_.typesize_sampled_wei]);
                            uni_vbroadcastss(xmm_v3,
                                             dword[aux_reg_sampled_wei + ind_off_lh * jcp_.typesize_sampled_wei]);
                            uni_vbroadcastss(xmm_v4,
                                             dword[aux_reg_sampled_wei + ind_off_hh * jcp_.typesize_sampled_wei]);

                            size_t input_buffer_off = static_cast<size_t>(kh) * jcp_.kw * jcp_.ic + kw * jcp_.ic;
                            uni_vpmovsxdq(xmm_v1_off, xmm_v1_off);
                            uni_vmovq(reg_tmp_64, xmm_v1_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);
                            checkZeroWei(xmm_v1, nullify_v1_tail);
                            uni_vmovss(xmm_v1, ptr[reg_tmp_64]);
                            uni_vmulss(xmm_v1, xmm_v1, xmm_w1);
                            jmp(nullify_v1_end_tail, T_NEAR);
                            L(nullify_v1_tail);
                            { uni_vpxor(xmm_v1, xmm_v1, xmm_v1); }
                            L(nullify_v1_end_tail);

                            uni_vpmovsxdq(xmm_v2_off, xmm_v2_off);
                            uni_vmovq(reg_tmp_64, xmm_v2_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);
                            checkZeroWei(xmm_v2, nullify_v2_tail);
                            uni_vmovss(xmm_v2, ptr[reg_tmp_64]);
                            uni_vmulss(xmm_v2, xmm_v2, xmm_w2);
                            jmp(nullify_v2_end_tail, T_NEAR);
                            L(nullify_v2_tail);
                            { uni_vpxor(xmm_v2, xmm_v2, xmm_v2); }
                            L(nullify_v2_end_tail);

                            uni_vpmovsxdq(xmm_v3_off, xmm_v3_off);
                            uni_vmovq(reg_tmp_64, xmm_v3_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);
                            checkZeroWei(xmm_v3, nullify_v3_tail);
                            uni_vmovss(xmm_v3, ptr[reg_tmp_64]);
                            uni_vmulss(xmm_v3, xmm_v3, xmm_w3);
                            jmp(nullify_v3_end_tail, T_NEAR);
                            L(nullify_v3_tail);
                            { uni_vpxor(xmm_v3, xmm_v3, xmm_v3); }
                            L(nullify_v3_end_tail);

                            uni_vpmovsxdq(xmm_v4_off, xmm_v4_off);
                            uni_vmovq(reg_tmp_64, xmm_v4_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);
                            checkZeroWei(xmm_v4, nullify_v4_tail);
                            uni_vmovss(xmm_v4, ptr[reg_tmp_64]);
                            uni_vmulss(xmm_v4, xmm_v4, xmm_w4);
                            jmp(nullify_v4_end_tail, T_NEAR);
                            L(nullify_v4_tail);
                            { uni_vpxor(xmm_v4, xmm_v4, xmm_v4); }
                            L(nullify_v4_end_tail);

                            uni_vaddss(xmm_v1, xmm_v1, xmm_v2);
                            uni_vaddss(xmm_v1, xmm_v1, xmm_v3);
                            uni_vaddss(xmm_v1, xmm_v1, xmm_v4);
                            uni_vmovss(ptr[aux3_reg_input_buffer + input_buffer_off * jcp_.typesize_in], xmm_v1);

                            add(aux2_reg_input, jcp_.typesize_in);
                            add(aux3_reg_input_buffer, jcp_.typesize_in);
                            sub(reg_ic_iter, 1);
                            jmp(ic_loop_tail, T_NEAR);
                        }
                        jmp(loop_end, T_NEAR);
                        L(loop_end);
                    }
                }
            }

            add(aux_reg_sampled_wei,
                sampledPointsPerPixel * jcp_.kh * jcp_.kw * jcp_.oh * jcp_.ow * jcp_.typesize_sampled_wei);
            add(aux_reg_sampled_offs,
                sampledPointsPerPixel * jcp_.kh * jcp_.kw * jcp_.oh * jcp_.ow * jcp_.typesize_sampled_offsets);
            add(aux_reg_input, ic_per_def_group * jcp_.typesize_in);
            add(aux2_reg_input_buffer, ic_per_def_group * jcp_.typesize_in);
            inc(reg_dg_iter);
            jmp(dg_loop, T_NEAR);
        }

        L(dg_loop_end);
        pop(reg_sampled_offs);
    }

    void store_output(int ow_step, int oc_blocks_step, int oc_step) {
        int repeats = isa == cpu::x64::sse41 && oc_step > (jcp_.oc_block / 2) ? 2 : 1;

        if (jcp_.with_bias) {
            for (int r = 0; r < repeats; r++) {
                for (int ocb = 0; ocb < oc_blocks_step; ocb++) {
                    size_t bias_off = static_cast<size_t>(ocb) * jcp_.oc_block + r * jcp_.oc_block / 2;
                    uni_vmovups(Vmm(0), ptr[aux_reg_bias + bias_off * jcp_.typesize_bia]);

                    for (int ow = 0; ow < ow_step; ow++) {
                        Vmm vmm_acc = get_vmm_acc(r * jcp_.ur_w * jcp_.nb_oc_blocking + ocb * ow_step + ow);
                        uni_vaddps(vmm_acc, vmm_acc, Vmm(0));
                    }
                }
            }
        }

        if (isa == avx512_core && oc_step != jcp_.oc_block) {
            int mask = (1 << oc_step) - 1;
            mov(reg_tmp_32, mask);
            kmovw(ktail_mask, reg_tmp_32);
        }

        for (int r = 0; r < repeats; r++) {
            int tail_size =
                isa == cpu::x64::sse41 ? std::min(jcp_.oc_block / 2, oc_step - r * jcp_.oc_block / 2) : oc_step;
            bool is_scalar_store = isa == cpu::x64::sse41 ? tail_size < jcp_.oc_block / 2 : tail_size < jcp_.oc_block;
            if (is_scalar_store) {
                for (int ow = 0; ow < ow_step; ow++) {
                    Vmm vmm_dst = get_vmm_acc(r * jcp_.ur_w * jcp_.nb_oc_blocking + ow);
                    Xmm xmm_dst = get_xmm_acc(r * jcp_.ur_w * jcp_.nb_oc_blocking + ow);

                    if (isa == avx512_core) {
                        size_t out_off = static_cast<size_t>(ow) * jcp_.oc;
                        uni_vmovups(ptr[aux_reg_output + out_off * jcp_.typesize_out], vmm_dst | ktail_mask);
                    } else {
                        for (int oc = 0; oc < tail_size; oc++) {
                            size_t out_off = static_cast<size_t>(ow) * jcp_.oc + oc + r * (jcp_.oc_block / 2);
                            uni_vmovq(reg_tmp_64, xmm_dst);
                            mov(ptr[aux_reg_output + out_off * jcp_.typesize_out], reg_tmp_32);

                            if (isa == cpu::x64::sse41) {
                                psrldq(vmm_dst, jcp_.typesize_out);
                            } else {
                                Ymm ymm_dst = get_ymm_acc(ow);
                                auto vmm_tmp = Vmm(0);
                                auto ymm_tmp = Ymm(0);

                                vperm2i128(ymm_tmp, ymm_dst, ymm_dst, 0x01);
                                vpalignr(ymm_dst, vmm_tmp, ymm_dst, jcp_.typesize_out);
                            }
                        }
                    }
                }
            } else {
                for (int ocb = 0; ocb < oc_blocks_step; ocb++) {
                    for (int ow = 0; ow < ow_step; ow++) {
                        Vmm vmm_acc = get_vmm_acc(r * jcp_.ur_w * jcp_.nb_oc_blocking + ocb * ow_step + ow);
                        size_t out_off = static_cast<size_t>(ow) * jcp_.oc * jcp_.ngroups + ocb * jcp_.oc_block +
                                         r * (jcp_.oc_block / 2);
                        uni_vmovups(ptr[aux_reg_output + out_off * jcp_.typesize_out], vmm_acc);
                    }
                }
            }
        }
    }

    void oc_loop(int ow_step) {
        Label oc_unrolled_loop;
        Label oc_main_loop;
        Label oc_tail;

        mov(aux_reg_input_buffer, reg_input_buffer);

        push(reg_output);
        push(reg_bias);
        push(reg_input);
        push(reg_kernel);

        prepare_buffer(ow_step);

        pop(reg_kernel);
        pop(reg_input);
        pop(reg_bias);
        pop(reg_output);

        push(reg_sampled_offs);
        push(reg_ow_pos);
        push(aux2_reg_kernel);

        mov(aux_reg_kernel, reg_kernel);
        mov(aux_reg_output, reg_output);
        mov(aux_reg_bias, reg_bias);
        mov(reg_oc_work, jcp_.oc);

        L(oc_unrolled_loop);
        {
            cmp(reg_oc_work, jcp_.nb_oc_blocking * jcp_.oc_block);
            jl(oc_main_loop, T_NEAR);

            ic_loop(ow_step, jcp_.nb_oc_blocking, jcp_.oc_block);
            store_output(ow_step, jcp_.nb_oc_blocking, jcp_.oc_block);

            add(aux_reg_kernel,
                jcp_.nb_oc_blocking * jcp_.nb_ic * jcp_.kh * jcp_.kw * jcp_.ic_block * jcp_.oc_block *
                    jcp_.typesize_in);
            add(aux_reg_output, jcp_.nb_oc_blocking * jcp_.oc_block * jcp_.typesize_out);
            add(aux_reg_bias, jcp_.nb_oc_blocking * jcp_.oc_block * jcp_.typesize_bia);
            sub(reg_oc_work, jcp_.nb_oc_blocking * jcp_.oc_block);

            jmp(oc_unrolled_loop, T_NEAR);
        }

        L(oc_main_loop);
        {
            cmp(reg_oc_work, jcp_.oc_block);
            jl(oc_tail, T_NEAR);

            ic_loop(ow_step, 1, jcp_.oc_block);
            store_output(ow_step, 1, jcp_.oc_block);

            add(aux_reg_kernel, jcp_.nb_ic * jcp_.kh * jcp_.kw * jcp_.ic_block * jcp_.oc_block * jcp_.typesize_in);
            add(aux_reg_output, jcp_.oc_block * jcp_.typesize_out);
            add(aux_reg_bias, jcp_.oc_block * jcp_.typesize_bia);
            sub(reg_oc_work, jcp_.oc_block);

            jmp(oc_main_loop, T_NEAR);
        }

        L(oc_tail);
        {
            if (jcp_.oc % jcp_.oc_block != 0) {
                ic_loop(ow_step, 1, jcp_.oc % jcp_.oc_block);
                store_output(ow_step, 1, jcp_.oc % jcp_.oc_block);
            }
        }

        pop(aux2_reg_kernel);
        pop(reg_ow_pos);
        pop(reg_sampled_offs);
    }
};
#endif
bool DeformableConvolution::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                                 std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                    ov::op::v1::DeformableConvolution::get_type_info_static(),
                    ov::op::v8::DeformableConvolution::get_type_info_static())) {
            errorMessage = "Node is not an instance of DeformableConvolution form the operation set v1 or v8.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

namespace {

struct DefConvKey {
    std::vector<std::shared_ptr<BlockedMemoryDesc>> descVector;
    DeformableConvolution::DefConvAttr defConvAttr;
    impl_desc_type implType;

    [[nodiscard]] size_t hash() const;
    bool operator==(const DefConvKey& rhs) const;
};

size_t DefConvKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    for (const auto& ptr : descVector) {
        if (ptr) {
            seed = get_vector_hash(seed, ptr->getBlockDims());
            seed = get_vector_hash(seed, ptr->getStrides());
            seed = get_vector_hash(seed, ptr->getOrder());
            seed = get_vector_hash(seed, ptr->getOffsetPaddingToData());
            seed = hash_combine(seed, ptr->getOffsetPadding());
        }
    }

    seed = get_vector_hash(seed, defConvAttr.stride);
    seed = get_vector_hash(seed, defConvAttr.dilation);
    seed = get_vector_hash(seed, defConvAttr.padL);

    seed = hash_combine(seed, implType);
    return seed;
}

bool DefConvKey::operator==(const DefConvKey& rhs) const {
    bool retVal = true;
    for (size_t i = 0; i < descVector.size(); i++) {
        if (descVector[i] != rhs.descVector[i]) {
            retVal = retVal && descVector[i] && rhs.descVector[i] &&
                     descVector[i]->getBlockDims() == rhs.descVector[i]->getBlockDims() &&
                     descVector[i]->getStrides() == rhs.descVector[i]->getStrides() &&
                     descVector[i]->getOrder() == rhs.descVector[i]->getOrder() &&
                     descVector[i]->getOffsetPaddingToData() == rhs.descVector[i]->getOffsetPaddingToData() &&
                     descVector[i]->getOffsetPadding() == rhs.descVector[i]->getOffsetPadding();
        }
    }

    retVal = retVal && defConvAttr.stride == rhs.defConvAttr.stride;
    retVal = retVal && defConvAttr.dilation == rhs.defConvAttr.dilation;
    retVal = retVal && defConvAttr.padL == rhs.defConvAttr.padL;

    retVal = retVal && implType == rhs.implType;
    return retVal;
}

}  // namespace

DeformableConvolution::DeformableConvolution(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    auto defConvNodeBase = ov::as_type_ptr<ov::op::util::DeformableConvolutionBase>(op);
    if (defConvNodeBase == nullptr) {
        THROW_CPU_NODE_ERR("is not an instance of DeformableConvolutionBase.");
    }

    defConvAttr.group = defConvNodeBase->get_group();
    defConvAttr.deformable_group = defConvNodeBase->get_deformable_group();
    auto& strides = defConvNodeBase->get_strides();
    for (uint64_t stride : strides) {
        defConvAttr.stride.push_back(stride);
    }

    auto& dilations = defConvNodeBase->get_dilations();
    for (uint64_t dilation : dilations) {
        defConvAttr.dilation.push_back(dilation - 1);
    }

    defConvAttr.padL = defConvNodeBase->get_pads_begin();

    autoPadding = one_of(defConvNodeBase->get_auto_pad(), ov::op::PadType::SAME_UPPER, ov::op::PadType::SAME_LOWER);

    if (op->get_type_info() == ov::op::v8::DeformableConvolution::get_type_info_static()) {
        auto defConvNode = ov::as_type_ptr<ov::op::v8::DeformableConvolution>(op);
        if (defConvNode == nullptr) {
            THROW_CPU_NODE_ERR("is not an instance of DeformableConvolution from opset8.");
        }
        defConvAttr.with_bilinear_pad = defConvNode->get_bilinear_interpolation_pad();
    } else {
        defConvAttr.with_bilinear_pad = false;
    }
}

void DeformableConvolution::getSupportedDescriptors() {
    if (getParentEdges().size() != 3 && getParentEdges().size() != 4) {
        THROW_CPU_NODE_ERR("has incorrect number of input edges");
    }
    if (getChildEdges().empty()) {
        THROW_CPU_NODE_ERR("has incorrect number of output edges");
    }
    if (getInputShapeAtPort(DATA_ID).getRank() != 4) {
        THROW_CPU_NODE_ERR("has unsupported mode. Only 4D blobs are supported as input.");
    }
    if (getInputShapeAtPort(OFF_ID).getRank() != 4) {
        THROW_CPU_NODE_ERR("doesn't support 1st input with rank: ", getInputShapeAtPort(OFF_ID).getRank());
    }
    if (getInputShapeAtPort(WEI_ID).getRank() != 4) {
        THROW_CPU_NODE_ERR("doesn't support 2nd input with rank: ", getInputShapeAtPort(WEI_ID).getRank());
    }
    if (getOutputShapeAtPort(DATA_ID).getRank() != 4) {
        THROW_CPU_NODE_ERR("doesn't support output with rank: ", getOutputShapeAtPort(DATA_ID).getRank());
    }
}

void DeformableConvolution::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    size_t inputsNumber = getOriginalInputsNumber();
    NodeConfig config;
    config.inConfs.resize(inputsNumber);
    config.inConfs[0].constant(false);
    config.inConfs[0].inPlace(-1);
    config.inConfs[1].constant(false);
    config.inConfs[1].inPlace(-1);
    config.inConfs[2].constant(false);
    config.inConfs[2].inPlace(-1);
    if (inputsNumber > 3) {
        config.inConfs[3].constant(false);
        config.inConfs[3].inPlace(-1);
    }

    config.outConfs.resize(1);
    config.outConfs[0].constant(false);
    config.outConfs[0].inPlace(-1);

    impl_desc_type impl_type = impl_desc_type::ref;
    const int simd_w = mayiuse(cpu::x64::avx512_core) ? 16 : 8;

    auto& weiDims = getInputShapeAtPort(WEI_ID).getDims();
    if (weiDims[1] == Shape::UNDEFINED_DIM || weiDims[0] == Shape::UNDEFINED_DIM ||
        // 1. strict fallback, until devising of multigroup handling in common case
        defConvAttr.group != 1 ||
        // 2. common fallback, except specific n_group / n_channel combinations
        (defConvAttr.group != 1 &&
         ((weiDims[1] % simd_w != 0)                                // in_channels_per_gr !% simd_w
          || ((weiDims[0] / defConvAttr.group) % simd_w != 0)))) {  // out_channels_per_gr !% simd_w
        enforceRef = true;
    } else {
        enforceRef = false;
    }

    if (enforceRef) {
        impl_type = impl_desc_type::ref;
    } else if (mayiuse(cpu::x64::avx512_core)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    }

    if (!enforceRef && mayiuse(cpu::x64::sse41)) {
        // optimized implementation
        auto dataFormat = memory::format_tag::nhwc;
        auto offFormat = memory::format_tag::nchw;
        auto weiFormat = mayiuse(avx512_core) ? memory::format_tag::OIhw16i16o : memory::format_tag::OIhw8i8o;
        config.inConfs[DATA_ID].setMemDesc(
            std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(DATA_ID), memory::data_type::f32, dataFormat));
        config.inConfs[OFF_ID].setMemDesc(
            std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(OFF_ID), memory::data_type::f32, offFormat));

        config.inConfs[WEI_ID].setMemDesc(
            std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(WEI_ID), memory::data_type::f32, weiFormat));

        if (inputsNumber > 3) {
            config.inConfs[MOD_ID].setMemDesc(std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(MOD_ID),
                                                                                      memory::data_type::f32,
                                                                                      memory::format_tag::nchw));
        }
        config.outConfs[0].setMemDesc(
            std::make_shared<DnnlBlockedMemoryDesc>(getOutputShapeAtPort(DATA_ID), memory::data_type::f32, dataFormat));
        supportedPrimitiveDescriptors.emplace_back(config, impl_type);
    } else {
        // reference implementation
        config.inConfs[DATA_ID].setMemDesc(std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(DATA_ID),
                                                                                   memory::data_type::f32,
                                                                                   memory::format_tag::nchw));
        config.inConfs[OFF_ID].setMemDesc(std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(OFF_ID),
                                                                                  memory::data_type::f32,
                                                                                  memory::format_tag::nchw));
        config.inConfs[WEI_ID].setMemDesc(std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(WEI_ID),
                                                                                  memory::data_type::f32,
                                                                                  memory::format_tag::oihw));
        if (inputsNumber > 3) {
            config.inConfs[MOD_ID].setMemDesc(std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(MOD_ID),
                                                                                      memory::data_type::f32,
                                                                                      memory::format_tag::nchw));
        }
        config.outConfs[0].setMemDesc(std::make_shared<DnnlBlockedMemoryDesc>(getOutputShapeAtPort(DATA_ID),
                                                                              memory::data_type::f32,
                                                                              memory::format_tag::nchw));
        supportedPrimitiveDescriptors.emplace_back(config, impl_type);
    }
}

void DeformableConvolution::DefConvExecutor::prepareSamplingWeights(const float* offsets,
                                                                    const float* modulation,
                                                                    bool enforceRef) {
    const int MB = jcp.mb;
    const int OH = jcp.oh;
    const int OW = jcp.ow;

    const int KH = jcp.kh;
    const int KW = jcp.kw;
    const int ker_size = KH * KW;

    const int DG = jcp.dg;

    const int IH = jcp.ih;
    const int IW = jcp.iw;

    const int KSH = jcp.stride_h;
    const int KSW = jcp.stride_w;

    const int KDH = jcp.dilate_h;
    const int KDW = jcp.dilate_w;

    const int padT = jcp.t_pad;
    const int padL = jcp.l_pad;

    const bool with_bi_pad = jcp.with_bi_pad;

    auto precompKer = [&](int mb, int dg, int oh, int ow) {
        int sampledCoordIndex = (mb * DG * OH * OW + dg * OH * OW + oh * OW + ow) * KH * KW * sampledPointsPerPixel;
        const int h_in = oh * KSH - padT;
        const int w_in = ow * KSW - padL;

        const float* data_offset_ptr = offsets + mb * offStrides[0] + (dg * 2 * KH * KW) * offStrides[1];
        const float* modulation_offset_ptr = nullptr;
        if (modulation != nullptr) {
            modulation_offset_ptr = modulation + mb * modStrides[0] + (dg * ker_size) * modStrides[1];
        }

        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                const size_t data_offset_h_index =
                    2 * (static_cast<size_t>(kh) * KW + kw) * offStrides[1] + oh * offStrides[2] + ow * offStrides[3];
                const size_t data_offset_w_index = (2 * (static_cast<size_t>(kh) * KW + kw) + 1) * offStrides[1] +
                                                   oh * offStrides[2] + ow * offStrides[3];
                const float offset_h = data_offset_ptr[data_offset_h_index];
                const float offset_w = data_offset_ptr[data_offset_w_index];
                float map_h = h_in + kh * (KDH + 1) + offset_h;
                float map_w = w_in + kw * (KDW + 1) + offset_w;
                bool skip_compute;
                if (with_bi_pad) {
                    skip_compute = !(static_cast<int>(map_w) > -1 && static_cast<int>(map_w) < IW &&
                                     static_cast<int>(map_h) > -1 && static_cast<int>(map_h) < IH);
                } else {
                    skip_compute = !(map_w >= 0 && map_w < IW && map_h >= 0 && map_h < IH);
                }
                if (!skip_compute) {
                    // modulations precomp.
                    float modulation_scalar = 1.0f;

                    if (modulation_offset_ptr != nullptr) {
                        size_t modulation_index =
                            (kh * KW + kw) * modStrides[1] + oh * modStrides[2] + ow * modStrides[3];
                        modulation_scalar = modulation_offset_ptr[modulation_index];
                    }
                    // interpolation precomp.
                    const int cur_h_end = IH;
                    const int cur_w_end = IW;
                    int h_low =
                        with_bi_pad ? static_cast<int>(floorf(map_h)) : std::max(static_cast<int>(floorf(map_h)), 0);
                    int w_low =
                        with_bi_pad ? static_cast<int>(floorf(map_w)) : std::max(static_cast<int>(floorf(map_w)), 0);
                    int h_high = with_bi_pad ? h_low + 1 : std::min(static_cast<int>(ceilf(map_h)), cur_h_end - 1);
                    int w_high = with_bi_pad ? w_low + 1 : std::min(static_cast<int>(ceilf(map_w)), cur_w_end - 1);

                    float lh = map_h - h_low;
                    float lw = map_w - w_low;
                    float hh = 1 - lh, hw = 1 - lw;

                    int h_ind_low = std::max(h_low, 0);
                    int h_ind_high = std::min(h_high, cur_h_end - 1);
                    int w_ind_low = std::max(w_low, 0);
                    int w_ind_high = std::min(w_high, cur_w_end - 1);

                    hh = (h_low >= 0 ? hh : 0);
                    hw = (w_low >= 0 ? hw : 0);
                    lh = (h_high < cur_h_end ? lh : 0);
                    lw = (w_high < cur_w_end ? lw : 0);

                    const int h_off_low = h_ind_low * (srcStrides[2] / srcStrides[3]);
                    const int h_off_high = h_ind_high * (srcStrides[2] / srcStrides[3]);
                    const int w_off_low = w_ind_low;
                    const int w_off_high = w_ind_high;
                    pSampledCoordsVector[sampledCoordIndex] = h_off_high + w_off_high;
                    pSampledCoordsVector[sampledCoordIndex + 1] = h_off_high + w_off_low;
                    pSampledCoordsVector[sampledCoordIndex + 2] = h_off_low + w_off_high;
                    pSampledCoordsVector[sampledCoordIndex + 3] = h_off_low + w_off_low;

                    float w22 = hh * hw * modulation_scalar, w21 = hh * lw * modulation_scalar,
                          w12 = lh * hw * modulation_scalar, w11 = lh * lw * modulation_scalar;

                    pInterpWeightsVector[sampledCoordIndex] = w11;
                    pInterpWeightsVector[sampledCoordIndex + 1] = w12;
                    pInterpWeightsVector[sampledCoordIndex + 2] = w21;
                    pInterpWeightsVector[sampledCoordIndex + 3] = w22;
                } else {
                    pSampledCoordsVector[sampledCoordIndex] = 0;
                    pInterpWeightsVector[sampledCoordIndex] = 0;
                    pInterpWeightsVector[sampledCoordIndex + 1] = 0;
                    pInterpWeightsVector[sampledCoordIndex + 2] = 0;
                    pInterpWeightsVector[sampledCoordIndex + 3] = 0;
                }
                sampledCoordIndex += sampledPointsPerPixel;
            }
        }
    };

    parallel_nd(MB, DG, OH, OW, [&](dim_t mb, dim_t dg, dim_t oh, dim_t ow) {
        precompKer(mb, dg, oh, ow);
    });
}

DeformableConvolution::DefConvExecutor::DefConvExecutor(
    const DefConvAttr& defConvAttr,
    const std::vector<std::shared_ptr<BlockedMemoryDesc>>& descVector) {
    if (descVector.size() != 4 && descVector.size() != 5) {
        OPENVINO_THROW("Deformable Convolution executor got incorrect desc's count (", descVector.size(), ")");
    }
    bool withModulation = descVector.size() == 5;

    auto& srcDesc = descVector[DATA_ID];
    auto& dstDesc = descVector[descVector.size() - 1];
    srcStrides = std::vector<size_t>(srcDesc->getStrides().size());
    offStrides = descVector[OFF_ID]->getStrides();
    weiStrides = descVector[WEI_ID]->getStrides();
    dstStrides = std::vector<size_t>(dstDesc->getStrides().size());
    pSampledCoordsVector = nullptr;
    pInterpWeightsVector = nullptr;
    for (size_t i = 0; i < srcDesc->getStrides().size(); i++) {
        srcStrides[srcDesc->getOrder()[i]] = srcDesc->getStrides()[i];
    }
    for (size_t i = 0; i < dstDesc->getStrides().size(); i++) {
        dstStrides[dstDesc->getOrder()[i]] = dstDesc->getStrides()[i];
    }

    if (withModulation) {
        modStrides = descVector[MOD_ID]->getStrides();
    }

    const VectorDims srcDims = descVector[DATA_ID]->getShape().getStaticDims();
    const VectorDims weiDims = descVector[WEI_ID]->getShape().getStaticDims();
    const VectorDims dstDims = descVector[descVector.size() - 1]->getShape().getStaticDims();

    jcp.dg = defConvAttr.deformable_group;
    jcp.ngroups = defConvAttr.group;

    jcp.mb = srcDims[0];

    jcp.oc = dstDims[1] / jcp.ngroups;
    jcp.ic = srcDims[1] / jcp.ngroups;

    jcp.ih = srcDims[2];
    jcp.iw = srcDims[3];
    jcp.oh = dstDims[2];
    jcp.ow = dstDims[3];

    jcp.kh = weiDims[2];
    jcp.kw = weiDims[3];

    jcp.t_pad = defConvAttr.padL[0];
    jcp.l_pad = defConvAttr.padL[1];

    jcp.stride_h = defConvAttr.stride[0];
    jcp.stride_w = defConvAttr.stride[1];

    jcp.dilate_h = defConvAttr.dilation[0];
    jcp.dilate_w = defConvAttr.dilation[1];

    jcp.with_bias = false;
    jcp.with_bi_pad = defConvAttr.with_bilinear_pad;
    jcp.with_modulation = withModulation;
    const int simd_w = mayiuse(cpu::x64::avx512_core) ? 16 : 8;
    jcp.ic_block = simd_w;
    jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);

    jcp.oc_block = simd_w;
    jcp.oc_padded = rnd_up(jcp.oc, jcp.oc_block);
    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);

    jcp.typesize_in = sizeof(float);
    jcp.typesize_off = sizeof(float);
    jcp.typesize_sampled_wei = sizeof(float);
    jcp.typesize_sampled_offsets = sizeof(int);
    jcp.typesize_out = sizeof(float);

    jcp.ur_w = mayiuse(cpu::x64::avx512_core) ? 6 : 3;
    jcp.nb_oc_blocking = !mayiuse(cpu::x64::avx2) ? 2 : 4;

    jcp.nthr = dnnl_get_max_threads();
}

DeformableConvolution::DefConvJitExecutor::DefConvJitExecutor(
    const DefConvAttr& defConvAttr,
    const std::vector<std::shared_ptr<BlockedMemoryDesc>>& descVector)
    : DefConvExecutor(defConvAttr, descVector) {
#if defined(OPENVINO_ARCH_X86_64)
    if (mayiuse(cpu::x64::avx512_core)) {
        def_conv_kernel = std::make_shared<jit_uni_def_conv_kernel_f32<cpu::x64::avx512_core>>(jcp);
    } else if (mayiuse(cpu::x64::avx2)) {
        def_conv_kernel = std::make_shared<jit_uni_def_conv_kernel_f32<cpu::x64::avx2>>(jcp);
    } else if (mayiuse(cpu::x64::sse41)) {
        def_conv_kernel = std::make_shared<jit_uni_def_conv_kernel_f32<cpu::x64::sse41>>(jcp);
    } else {
        OPENVINO_THROW("Can't create DefConvJitExecutor");
    }
    if (def_conv_kernel) {
        def_conv_kernel->create_ker();
    } else {
        OPENVINO_THROW("Can't compile DefConvJitExecutor");
    }
#endif
}

void DeformableConvolution::DefConvRefExecutor::exec(const float* src,
                                                     const float* offsets,
                                                     const float* weights,
                                                     const float* modulation,
                                                     float* dst,
                                                     int* pSampledCoordsVector,
                                                     float* pInterpWeightsVector) {
    this->pSampledCoordsVector = pSampledCoordsVector;
    this->pInterpWeightsVector = pInterpWeightsVector;
    prepareSamplingWeights(offsets, modulation, true);
    const int G = jcp.ngroups;
    const int MB = jcp.mb;
    const int OH = jcp.oh;
    const int OW = jcp.ow;

    const int OC = jcp.oc;
    const int IC = jcp.ic;
    const int KH = jcp.kh;
    const int KW = jcp.kw;
    const int ker_size = KH * KW;

    const int DG = jcp.dg;

    const int DGHW = DG * OH * OW;
    const int HW = OH * OW;

    const int channel_per_deformable_group = (IC * G) / DG;
    const size_t group_wei_stride = weiStrides[0] * OC;
    auto compKer = [OV_CAPTURE_CPY_AND_THIS](int g, int mb, int oc, int oh, int ow) {
        float d = 0;
        for (int ic = 0; ic < IC; ic++) {
            const float* data_im_ptr = src + mb * srcStrides[0] + (g * IC + ic) * srcStrides[1];
            const int deformable_group_index = (IC * g + ic) / channel_per_deformable_group;
            int sampledCoordIndex =
                (mb * DGHW + deformable_group_index * HW + oh * OW + ow) * ker_size * sampledPointsPerPixel;
            size_t weiIndex = static_cast<size_t>(g) * group_wei_stride + oc * weiStrides[0] + ic * weiStrides[1];
            for (size_t kh_off = 0; kh_off < KH * weiStrides[2]; kh_off += weiStrides[2]) {
                for (size_t kw_off = 0; kw_off < KW * weiStrides[3]; kw_off += weiStrides[3]) {
                    // check if current addendum marked as equal zero
                    if (pSampledCoordsVector[sampledCoordIndex] != -1) {
                        const int v11 = pSampledCoordsVector[sampledCoordIndex];
                        const int v12 = pSampledCoordsVector[sampledCoordIndex + 1];
                        const int v21 = pSampledCoordsVector[sampledCoordIndex + 2];
                        const int v22 = pSampledCoordsVector[sampledCoordIndex + 3];

                        float val = 0;
                        float w11 = pInterpWeightsVector[sampledCoordIndex++];
                        float w12 = pInterpWeightsVector[sampledCoordIndex++];
                        float w21 = pInterpWeightsVector[sampledCoordIndex++];
                        float w22 = pInterpWeightsVector[sampledCoordIndex++];

                        // Prevent access to invalid memory in the case, when
                        // data_im_ptr[v_i1_i2] is out of the input memory.
                        // Logic of skipping of such points realized by nullifying
                        // of corresponding weight, but we must explicitly check it, because
                        // 0 * (*wrong_pointer) != 0 in common case, i.e.
                        // 0 * NaN == NaN or throws segfault
                        val += ((w11 == 0) ? 0 : w11 * data_im_ptr[v11]);
                        val += ((w12 == 0) ? 0 : w12 * data_im_ptr[v12]);
                        val += ((w21 == 0) ? 0 : w21 * data_im_ptr[v21]);
                        val += ((w22 == 0) ? 0 : w22 * data_im_ptr[v22]);

                        d += val * weights[weiIndex + kh_off + kw_off];
                    } else {
                        sampledCoordIndex += sampledPointsPerPixel;
                    }
                }
            }
        }
        return d;
    };

    parallel_nd(G, MB, OC, OH, OW, [&](dnnl_dim_t g, dnnl_dim_t mb, dnnl_dim_t oc, dnnl_dim_t oh, dnnl_dim_t ow) {
        dst[mb * dstStrides[0] + (g * OC + oc) * dstStrides[1] + oh * dstStrides[2] + ow * dstStrides[3]] =
            compKer(g, mb, oc, oh, ow);
    });
}

void DeformableConvolution::prepareParams() {
    auto dstMemPtr = getDstMemoryAtPort(0);
    auto srcMemPtr = getSrcMemoryAtPort(DATA_ID);
    auto offMemPtr = getSrcMemoryAtPort(OFF_ID);
    auto weiMemPtr = getSrcMemoryAtPort(WEI_ID);

    if (!dstMemPtr || !dstMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("has undefined destination memory");
    }
    if (!srcMemPtr || !srcMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("has undefined input memory");
    }
    if (!offMemPtr || !offMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("has undefined offsets shape memory");
    }
    if (!weiMemPtr || !weiMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("has undefined weights memory");
    }

    if (getOriginalInputsNumber() > 3) {
        auto modMemPtr = getSrcMemoryAtPort(MOD_ID);
        if (!modMemPtr || !modMemPtr->isDefined()) {
            THROW_CPU_NODE_ERR("has undefined modulations memory");
        }
    }

    auto selectedPrimitiveDescriptor = getSelectedPrimitiveDescriptor();
    if (!selectedPrimitiveDescriptor) {
        THROW_CPU_NODE_ERR("doesn't have primitive descriptors.");
    }
    auto config = selectedPrimitiveDescriptor->getConfig();

    bool withModulation = getParentEdges().size() > 3;

    updatePadding();

    std::vector<std::shared_ptr<BlockedMemoryDesc>> descVector{
        getParentEdgeAt(DATA_ID)->getMemory().getDescWithType<BlockedMemoryDesc>(),
        getParentEdgeAt(OFF_ID)->getMemory().getDescWithType<BlockedMemoryDesc>(),
        getParentEdgeAt(WEI_ID)->getMemory().getDescWithType<BlockedMemoryDesc>()};

    if (withModulation) {
        descVector.push_back(getParentEdgeAt(MOD_ID)->getMemory().getDescWithType<BlockedMemoryDesc>());
    }
    descVector.push_back(getChildEdgeAt(0)->getMemory().getDescWithType<BlockedMemoryDesc>());

    DefConvKey key = {descVector, defConvAttr, getSelectedPrimitiveDescriptor()->getImplementationType()};

    const int MB = getParentEdgeAt(DATA_ID)->getMemory().getStaticDims()[0];
    const int OH = getChildEdgeAt(0)->getMemory().getStaticDims()[2];
    const int OW = getChildEdgeAt(0)->getMemory().getStaticDims()[3];

    const int KH = getParentEdgeAt(WEI_ID)->getMemory().getStaticDims()[2];
    const int KW = getParentEdgeAt(WEI_ID)->getMemory().getStaticDims()[3];

    const int DG = defConvAttr.deformable_group;

    // allocate sampling weights and indices
    sampledCoordsVector.resize(MB * DG * KH * KW * OH * OW * sampledPointsPerPixel);
    interpWeightsVector.resize(MB * DG * KH * KW * OH * OW * sampledPointsPerPixel);

    execPtr = nullptr;

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, [](const DefConvKey& key) -> std::shared_ptr<DefConvExecutor> {
        if (key.implType == impl_desc_type::ref) {
            return std::make_shared<DefConvRefExecutor>(key.defConvAttr, key.descVector);
        }
        return std::make_shared<DefConvJitExecutor>(key.defConvAttr, key.descVector);
    });
    execPtr = result.first;

    if (!execPtr) {
        THROW_CPU_NODE_ERR("Primitive descriptor was not found.");
    }
}

void DeformableConvolution::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void DeformableConvolution::DefConvJitExecutor::exec(const float* src,
                                                     const float* offsets,
                                                     const float* weights,
                                                     const float* modulation,
                                                     float* dst,
                                                     int* pSampledCoordsVector,
                                                     float* pInterpWeightsVector) {
    this->pSampledCoordsVector = pSampledCoordsVector;
    this->pInterpWeightsVector = pInterpWeightsVector;
    prepareSamplingWeights(offsets, modulation, false);
    size_t buffer_size = static_cast<size_t>(jcp.nthr) * jcp.ur_w * jcp.kh * jcp.kw * jcp.ic * jcp.typesize_in;
    std::vector<float> input_buffer(buffer_size, 0);
    float* input_buffer_ptr = input_buffer.data();

    parallel_for3d(jcp.mb, jcp.ngroups, jcp.oh, [&](size_t n, size_t g, size_t oh) {
        auto ithr = parallel_get_thread_num();

        auto par_conv = jit_def_conv_call_args();

        const size_t _oc = g * jcp.nb_oc;
        const size_t _ic = g * jcp.nb_ic;

        par_conv.src = &src[n * srcStrides[0] + _ic * jcp.ic_block * srcStrides[1]];
        par_conv.sampledWei =
            &(pInterpWeightsVector[(n * jcp.dg * jcp.oh + oh) * jcp.kh * jcp.kw * jcp.ow * sampledPointsPerPixel]);
        par_conv.sampledCoords =
            &(pSampledCoordsVector[(n * jcp.dg * jcp.oh + oh) * jcp.kh * jcp.kw * jcp.ow * sampledPointsPerPixel]);
        par_conv.filt = &weights[g * jcp.nb_oc * jcp.nb_ic * jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block];
        par_conv.dst = &dst[n * dstStrides[0] + _oc * jcp.oc_block * dstStrides[1] + oh * dstStrides[2]];
        par_conv.buf = input_buffer_ptr + ithr * jcp.ur_w * jcp.kh * jcp.kw * jcp.ic;

        par_conv.oh_pos = oh;

        (*def_conv_kernel)(&par_conv);
    });
}

void DeformableConvolution::execute(const dnnl::stream& strm) {
    const size_t inputsNumber = getOriginalInputsNumber();

    auto& srcMemory0 = getParentEdgeAt(0)->getMemory();
    auto& srcMemory1 = getParentEdgeAt(1)->getMemory();
    auto& srcMemory2 = getParentEdgeAt(2)->getMemory();
    auto& dstMemory = getChildEdgeAt(0)->getMemory();

    const auto* src = srcMemory0.getDataAs<const float>();
    const auto* offsets = srcMemory1.getDataAs<const float>();
    const auto* weights = srcMemory2.getDataAs<const float>();
    float* modulation = nullptr;
    if (inputsNumber > 3) {
        modulation = getSrcDataAtPortAs<float>(3);
    }

    auto* dst = dstMemory.getDataAs<float>();

    auto selectedPrimitiveDescriptor = getSelectedPrimitiveDescriptor();
    if (!selectedPrimitiveDescriptor) {
        THROW_CPU_NODE_ERR("doesn't have primitive descriptors.");
    }
    auto config = selectedPrimitiveDescriptor->getConfig();

    if (execPtr) {
        execPtr->exec(src, offsets, weights, modulation, dst, sampledCoordsVector.data(), interpWeightsVector.data());
    } else {
        THROW_CPU_NODE_ERR("executor doesn't exist");
    }
}

void DeformableConvolution::updatePadding() {
    if (isDynamicNode() && autoPadding) {
        defConvAttr.padL = shapeInference->get_pads_begin();
    }
}

bool DeformableConvolution::created() const {
    return getType() == Type::DeformableConvolution;
}

ov::element::Type DeformableConvolution::getRuntimePrecision() const {
    return getMaxPrecision(getInputPrecisions());
}

}  // namespace ov::intel_cpu::node
