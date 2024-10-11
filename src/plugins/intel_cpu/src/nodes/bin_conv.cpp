// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bin_conv.h"
#include "eltwise.h"
#include "fake_quantize.h"
#include "conv.h"
#include <memory>
#include <string>
#include <vector>
#include "dnnl_types.h"
#include "dnnl_extension_utils.h"
#include "openvino/core/parallel.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_depthwise_injector.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "utils/general_utils.h"
#include "openvino/opsets/opset1.hpp"
#include "utils/ngraph_utils.hpp"

// WA for xbyak.h
#ifdef _WIN32
# ifndef _WINSOCKAPI_
#  define _WINSOCKAPI_
# endif
# ifndef _WINSOCK2API_
#  define _WINSOCK2API_
# endif
#endif


using namespace dnnl;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {
namespace node {
#if defined(OPENVINO_ARCH_X86_64)
#define GET_OFF(field) offsetof(jit_bin_conv_call_args, field)

template <cpu_isa_t isa>
struct jit_uni_bin_conv_kernel_f32 : public jit_uni_bin_conv_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_bin_conv_kernel_f32)

    explicit jit_uni_bin_conv_kernel_f32(jit_bin_conv_params jcp, jit_dw_conv_params jcp_dw_conv, const dnnl_primitive_attr &attr) :
        jit_uni_bin_conv_kernel(jcp, jcp_dw_conv, attr), jit_generator(jit_name())  {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        const auto &p = attr_.post_ops_;
        int end_idx = jcp_.with_dw_conv ? p.find(primitive_kind::convolution) : p.len();
        for (int i = 0; i < end_idx; i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors.push_back(std::make_shared<jit_uni_eltwise_injector_f32<isa>>(
                        this, post_op.eltwise, true, eltwise_reserved, mask_post_op_reserved));
            } else if (post_op.is_depthwise()) {
                depthwise_injectors.push_back(std::make_shared<jit_uni_depthwise_injector_f32<isa>>(
                        this, post_op, mask_post_op_reserved));
            }
        }

        this->preamble();

        mov(reg_input_base, ptr[this->param1 + GET_OFF(src)]);
        mov(reg_output_base, ptr[this->param1 + GET_OFF(dst)]);
        mov(reg_kernel_base, ptr[this->param1 + GET_OFF(filt)]);

        mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
        mov(reg_oc_work, ptr[this->param1 + GET_OFF(oc_work)]);
        mov(reg_post_ops_data, ptr[this->param1 + GET_OFF(post_op_data)]);

        mov(reg_oc_off,  ptr[param1 + GET_OFF(oc_off)]);
        mov(reg_table, l_table);

        Label main_loop_label;
        Label tail_label;
        Label exit_label;

        cmp(reg_oc_work, jcp_.nb_oc_blocking * jcp_.oc_block);
        jne(main_loop_label, T_NEAR);

        solve_common(jcp_.nb_oc_blocking, jcp_.oc_block);

        sub(reg_oc_work, jcp_.nb_oc_blocking * jcp_.oc_block);

        jmp(exit_label, T_NEAR);

        int nbits = 8;

        L(main_loop_label); {
            cmp(reg_oc_work, jcp_.oc_block);
            jl(tail_label, T_NEAR);

            solve_common(1, jcp_.oc_block);

            sub(reg_oc_work, jcp_.oc_block);
            add(reg_kernel_base, jcp_.oc_block * jcp_.nb_ic * jcp_.kh * jcp_.kw * div_up(jcp_.ic_block, nbits) * jcp_.typesize_in);

            if (jcp_.with_dw_conv) {
                add(reg_output_base, jcp_.oc_block * jcp_dw_conv_.kh * jcp_.ow * jcp_.typesize_out);
            } else {
                if (jcp_.with_binarization)
                    add(reg_output_base, div_up(jcp_.oc_block, nbits) * jcp_.typesize_out);
                else
                    add(reg_output_base, jcp_.oc_block * jcp_.typesize_out);
            }

            add(reg_oc_off, jcp_.oc_block * sizeof(float));

            jmp(main_loop_label, T_NEAR);
        }

        L(tail_label);

        if (jcp_.oc % jcp_.oc_block != 0)
            solve_common(1, jcp_.oc % jcp_.oc_block);

        L(exit_label);

        this->postamble();

        prepare_table();

        for (auto& inj : eltwise_injectors)
            inj->prepare_table();
    }

private:
    using Vmm = typename conditional3<isa == x64::sse41, Xbyak::Xmm, isa == x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;

    using Ymm = const Xbyak::Ymm;
    using reg8_t = const Xbyak::Reg8;
    using reg16_t = const Xbyak::Reg16;
    using reg32_t = const Xbyak::Reg32;
    using reg64_t = const Xbyak::Reg64;

    reg64_t reg_input = r13;
    reg64_t reg_output = rbp;
    reg64_t reg_input_base = rax;
    reg64_t aux_reg_input = r8;
    reg64_t reg_kernel_base = rdx;
    reg64_t aux_reg_kernel = r9;
    reg64_t reg_output_base = rsi;
    reg64_t aux1_reg_input = reg_input_base;
    reg64_t aux1_reg_kernel = reg_output_base;

    reg64_t kj = r10;
    reg64_t oi_iter = r11;
    reg64_t reg_kh = abi_not_param1;
    reg64_t reg_overflow = reg_kh;
    reg64_t reg_oc_work = r14;
    reg64_t reg_table = r15;
    reg64_t reg_icb_iter = reg_oc_work;

    reg8_t reg_tmp_8 = r12b;
    reg16_t reg_tmp_16 = r12w;
    reg32_t reg_tmp_32 = r12d;
    reg64_t reg_tmp_64 = r12;

    reg64_t reg_d_weights = aux_reg_input;
    reg64_t reg_d_bias = aux_reg_kernel;
    reg64_t reg_oc_off = kj;
    reg64_t reg_post_ops_data = rbx;
    reg64_t reg_tmp2_64 = reg_oc_off;
    reg32_t reg_tmp2_32 = reg_oc_off.cvt32();

    reg64_t reg_b_weights = aux_reg_input;
    reg64_t reg_b_mask = aux_reg_kernel;
    reg64_t reg_b_out_mask = reg_icb_iter;

    reg64_t reg_shift = aux_reg_input;

    Vmm vmm_scale = Vmm(isa == x64::avx512_core ? 30 : 14);
    Vmm vmm_shift = Vmm(0);
    Vmm vmm_sum = Vmm(isa == x64::avx512_core ? 26 : 10);
    Vmm vmm_lookup = Vmm(isa == x64::avx512_core ? 28 : 12);
    Vmm vmm_mask = Vmm(isa == x64::avx512_core ? 29 : 13);
    Vmm vmm_one_u8 = Vmm(isa == x64::avx512_core ? 30 : 14);
    Vmm vmm_one_s16 = Vmm(isa == x64::avx512_core ? 31 : 15);
    Ymm ymm_tmp = Ymm(isa == x64::avx512_core ? 26 : 10);
    Vmm vmm_tmp = Vmm(isa == x64::avx512_core ? 26 : 10);
    Vmm vmm_tmp1 = Vmm(isa == x64::avx512_core ? 27 : 11);
    Vmm vmm_src = Vmm(0);
    Vmm vmm_tmp2 = Vmm(isa == x64::avx512_core ? 25 : 9);
    Vmm vmm_thr = Vmm(isa == x64::avx512_core ? 26 : 10);
    Vmm vmm_out_mask = Vmm(isa == x64::avx512_core ? 30 : 14);

    const unsigned char _cmp_gt_os = 6;

    Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);
    Xbyak::Opmask bin_mask0 = Xbyak::Opmask(5);
    Xbyak::Opmask bin_mask1 = Xbyak::Opmask(6);
    Xbyak::Opmask mask_post_op_reserved = Xbyak::Opmask(1);
    Xbyak::Reg64 eltwise_reserved = rax;

    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Label l_table;

    nstl::vector<std::shared_ptr<jit_uni_eltwise_injector_f32<isa>>> eltwise_injectors;
    nstl::vector<std::shared_ptr<jit_uni_depthwise_injector_f32<isa>>> depthwise_injectors;

    void cvt2ps(dnnl::memory::data_type type_in, Vmm vmm_in, const Xbyak::Operand &op, bool scalar_load) {
        Xmm xmm_in = Xmm(vmm_in.getIdx());

        switch (type_in) {
            case memory::data_type::f32:
            case memory::data_type::s32:
                if (scalar_load) {
                    mov(reg_tmp_32, op);
                    uni_vmovq(xmm_in, reg_tmp_64);
                } else {
                    uni_vmovups(vmm_in, op);
                }
                break;
            case memory::data_type::s8:
                if (scalar_load) {
                    movsx(reg_tmp_32, op);
                    uni_vmovq(xmm_in, reg_tmp_64);
                } else {
                    uni_vpmovsxbd(vmm_in, op);
                }
                break;
            case memory::data_type::u8:
                if (scalar_load) {
                    movzx(reg_tmp_32, op);
                    uni_vmovq(xmm_in, reg_tmp_64);
                } else {
                    uni_vpmovzxbd(vmm_in, op);
                }
                break;
            default: assert(!"unsupported data type");
        }

        if (type_in != data_type::f32)
            uni_vcvtdq2ps(vmm_in, vmm_in);
    }

    void store_dst(const Xbyak::Address &op, Vmm vmm_dst, bool scalar_store) {
        Ymm ymm_dst = Ymm(vmm_dst.getIdx());
        Xmm xmm_dst = Xmm(vmm_dst.getIdx());

        switch (jcp_.dst_dt) {
            case memory::data_type::f32:
            case memory::data_type::s32:
                if (scalar_store) {
                    movq(reg_tmp_64, xmm_dst);
                    mov(op, reg_tmp_32);
                } else {
                    uni_vmovups(op, vmm_dst);
                }
                break;
            case memory::data_type::s8:
                uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);

                if (isa != x64::sse41 && !scalar_store)
                    vpermq(ymm_dst, ymm_dst, 0x08);

                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);

                if (scalar_store) {
                    movq(reg_tmp_64, xmm_dst);
                    mov(op, reg_tmp_8);
                } else {
                    if (isa != x64::sse41)
                        vmovq(op, xmm_dst);
                    else
                        movd(op, xmm_dst);
                }
                break;
            case memory::data_type::u8:
            case memory::data_type::bin:
                uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);

                if (isa != x64::sse41 && !scalar_store)
                    vpermq(ymm_dst, ymm_dst, 0x08);

                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);

                if (scalar_store) {
                    movq(reg_tmp_64, xmm_dst);
                    mov(op, reg_tmp_8);
                } else {
                    if (isa != x64::sse41)
                        vmovq(op, xmm_dst);
                    else
                        movd(op, xmm_dst);
                }

                break;
            default:
                assert(!"unknown dst_dt");
        }
    }

    void apply_filter(int ur_w, int pad_l, int pad_r, int oc_blocks, int oc_step, int ic_blocks, bool last_icb, bool h_padded) {
        int kw = jcp_.kw;
        int kh = jcp_.kh;
        int stride_w = jcp_.stride_w;
        int dilate_w = jcp_.dilate_w + 1;
        int ic_blk = jcp_.ic_block;
        int oc_blk = jcp_.oc_block;

        int repeats = isa == x64::sse41 && oc_step > (oc_blk / 2) ? 2 : 1;
        int nbits = 8;

        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, div_up(pad_l - ki * dilate_w, stride_w));
            int jj_end = ur_w  - nstl::max(0, div_up(ki*dilate_w+pad_r-(kw-1)*dilate_w, stride_w));

            int _start = (!jcp_.exclude_pad) ? 0 : jj_start;
            int _end = (!jcp_.exclude_pad) ? ur_w : jj_end;

            for (int ifm2 = 0; ifm2 < ic_blocks; ifm2++) {
                for (int jj = _start; jj < _end; jj++) {
                    int inp_off = ((ki*dilate_w + jj*stride_w - pad_l)*div_up(jcp_.ic, nbits) +
                                   ifm2 * div_up(ic_blk, nbits)) * jcp_.typesize_in;

                    if (h_padded || jj < jj_start || jj >= jj_end) {
                        uni_vmovups(vmm_src, ptr[reg_table + 8 * vlen]);
                    } else {
                        uni_vpbroadcastd(vmm_src, ptr[aux1_reg_input + inp_off]);
                    }

                    for (int r = 0; r < repeats; r++) {
                        for (int ii = 0; ii < oc_blocks; ii++) {
                            int ker_off = (ifm2 * kh * kw * div_up(ic_blk, nbits) * oc_blk
                                           + ii * jcp_.nb_ic * div_up(ic_blk, nbits) * kh * kw * oc_blk
                                           + ki * div_up(ic_blk, nbits) * oc_blk
                                           + r * div_up(ic_blk, nbits) * (oc_blk / 2)) * jcp_.typesize_in;

                            uni_vmovups(vmm_tmp, ptr[aux1_reg_kernel + ker_off]);

                            uni_vpxor(vmm_tmp, vmm_tmp, vmm_src);
                            if (jcp_.ic_padded != jcp_.ic && last_icb && ifm2 == (ic_blocks - 1))
                                uni_vandps(vmm_tmp, vmm_tmp, ptr[reg_table + 7 * vlen]);

                            if (mayiuse(x64::avx512_vpopcnt)) {
                                vpopcntd(vmm_tmp, vmm_tmp);
                                uni_vpaddd(Vmm(1 + r * jcp_.ur_w * jcp_.nb_oc_blocking + ur_w * ii + jj),
                                           Vmm(1 + r * jcp_.ur_w * jcp_.nb_oc_blocking + ur_w * ii + jj), vmm_tmp);
                            } else {
                                if (isa == x64::sse41) {
                                    movups(vmm_tmp1, vmm_tmp);
                                    pand(vmm_tmp1, vmm_mask);
                                } else {
                                    uni_vandps(vmm_tmp1, vmm_mask, vmm_tmp);
                                }

                                uni_vpsrld(vmm_tmp, vmm_tmp, 4);
                                uni_vandps(vmm_tmp, vmm_tmp, vmm_mask);

                                if (isa == x64::sse41) {
                                    movups(vmm_tmp2, vmm_lookup);
                                    pshufb(vmm_tmp2, vmm_tmp);
                                    movups(vmm_tmp, vmm_lookup);
                                    pshufb(vmm_tmp, vmm_tmp1);
                                    paddb(vmm_tmp, vmm_tmp2);
                                } else {
                                    uni_vpshufb(vmm_tmp, vmm_lookup, vmm_tmp);
                                    uni_vpshufb(vmm_tmp1, vmm_lookup, vmm_tmp1);
                                    uni_vpaddb(vmm_tmp, vmm_tmp, vmm_tmp1);
                                }

                                if (mayiuse(avx512_core_vnni)) {
                                    vpdpbusd(Vmm(1 + r * jcp_.ur_w * jcp_.nb_oc_blocking + ur_w * ii + jj), vmm_tmp, vmm_one_u8);
                                } else {
                                    uni_vpmaddubsw(vmm_tmp, vmm_tmp, vmm_one_u8);
                                    uni_vpmaddwd(vmm_tmp, vmm_tmp, vmm_one_s16);
                                    uni_vpaddd(Vmm(1 + r * jcp_.ur_w * jcp_.nb_oc_blocking + ur_w * ii + jj),
                                               Vmm(1 + r * jcp_.ur_w * jcp_.nb_oc_blocking + ur_w * ii + jj), vmm_tmp);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void oh_step_unroll_kw(int ur_w, int pad_l, int pad_r, int oc_blocks, int oc_step, bool h_padded) {
        int kh = jcp_.kh;
        int kw = jcp_.kw;

        int nbits = 8;
        int inp_mult = div_up(jcp_.ic_block, nbits);
        int out_mult = jcp_.oc_block;

        Label icb_main_loop;
        Label icb_tail;

        mov(aux1_reg_input, aux_reg_input);
        mov(aux1_reg_kernel, aux_reg_kernel);

        mov(reg_icb_iter, jcp_.nb_ic);
        L(icb_main_loop);
        {
            cmp(reg_icb_iter, 1);
            jle(icb_tail, T_NEAR);

            apply_filter(ur_w, pad_l, pad_r, oc_blocks, oc_step, 1, false, h_padded);

            add(aux1_reg_input, inp_mult * jcp_.typesize_in);
            add(aux1_reg_kernel, kh * kw * inp_mult * out_mult * jcp_.typesize_in);
            sub(reg_icb_iter, 1);
            jmp(icb_main_loop, T_NEAR);
        }

        L(icb_tail);

        apply_filter(ur_w, pad_l, pad_r, oc_blocks, oc_step, 1, true, h_padded);
    }

    void kh_loop(int ur_w, int pad_l, int pad_r, int oc_blocks, int oc_step) {
        int iw = jcp_.iw;
        int kw = jcp_.kw;
        int dilate_h = jcp_.dilate_h + 1;

        int nbits = 8;
        const int inp_mult = dilate_h * div_up(jcp_.ic, nbits);

        Label t_overflow_label, no_t_overflow_label,
                b_overflow_label, no_b_overflow_label;

        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel_base);

        uni_vmovups(vmm_lookup,  ptr[reg_table + 0 * vlen]);
        uni_vmovups(vmm_mask,    ptr[reg_table + 1 * vlen]);
        uni_vmovups(vmm_one_u8,  ptr[reg_table + 5 * vlen]);
        uni_vmovups(vmm_one_s16, ptr[reg_table + 6 * vlen]);

        if (!jcp_.exclude_pad) {
            mov(reg_overflow,  ptr[param1 + GET_OFF(t_overflow)]);
            cmp(reg_overflow, 0);
            je(no_t_overflow_label, T_NEAR);
            L(t_overflow_label); {
                oh_step_unroll_kw(ur_w, pad_l, pad_r, oc_blocks, oc_step, true);

                add(aux_reg_kernel, jcp_.typesize_in * kw * jcp_.oc_block * div_up(jcp_.ic_block, nbits));
                dec(reg_overflow);
                cmp(reg_overflow, 0);
                jg(t_overflow_label, T_NEAR);
            }
            L(no_t_overflow_label);
        }

        Label skip_kh_loop;
        mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
        if (!jcp_.exclude_pad || (jcp_.exclude_pad &&
                                  (jcp_.kh - 1) * (jcp_.dilate_h + 1) < nstl::max(jcp_.t_pad, jcp_.b_pad))) {
            cmp(reg_kh, 0);
            je(skip_kh_loop, T_NEAR);
        }

        Label kh_label;
        L(kh_label);
        {
            oh_step_unroll_kw(ur_w, pad_l, pad_r, oc_blocks, oc_step, false);

            add(aux_reg_kernel, jcp_.typesize_in * kw * jcp_.oc_block * div_up(jcp_.ic_block, nbits));
            add(aux_reg_input, jcp_.typesize_in * iw * inp_mult);

            dec(reg_kh);
            cmp(reg_kh, 0);
            jg(kh_label, T_NEAR);
        }

        L(skip_kh_loop);

        if (!jcp_.exclude_pad) {
            mov(reg_overflow,  ptr[param1 + GET_OFF(b_overflow)]);
            cmp(reg_overflow, 0);
            je(no_b_overflow_label, T_NEAR);
            L(b_overflow_label); {
                oh_step_unroll_kw(ur_w, pad_l, pad_r, oc_blocks, oc_step, true);

                add(aux_reg_kernel, jcp_.typesize_in * kw * jcp_.oc_block * div_up(jcp_.ic_block, nbits));
                dec(reg_overflow);
                cmp(reg_overflow, 0);
                jg(b_overflow_label, T_NEAR);
            }
            L(no_b_overflow_label);
        }
    }

    void width_blk_step(int ur_w, int pad_l, int pad_r, int oc_blocks, int oc_step) {
        int nbits = 8;
        int repeats = isa == x64::sse41 && oc_step > (jcp_.oc_block / 2) ? 2 : 1;

        for (int r = 0; r < repeats; r++)
            for (int ii = 0; ii < oc_blocks; ii++)
                for (int jj = 0; jj < ur_w; jj++)
                    uni_vpxor(Vmm(1 + r * jcp_.ur_w * jcp_.nb_oc_blocking + ur_w * ii + jj),
                              Vmm(1 + r * jcp_.ur_w * jcp_.nb_oc_blocking + ur_w * ii + jj),
                              Vmm(1 + r * jcp_.ur_w * jcp_.nb_oc_blocking + ur_w * ii + jj));

        kh_loop(ur_w, pad_l, pad_r, oc_blocks, oc_step);

        if (isa == x64::avx512_core && oc_step != jcp_.oc_block) {
            int mask = (1 << oc_step) - 1;
            mov(reg_tmp_32, mask);
            kmovw(ktail_mask, reg_tmp_32);
        }

        const auto &p = attr_.post_ops_;
        for (int r = 0; r < repeats; r++) {
            int tail_size = isa == x64::sse41 ? nstl::min(jcp_.oc_block / 2, oc_step - r * jcp_.oc_block / 2) : oc_step;
            bool is_scalar_store = isa == x64::sse41 ? tail_size < jcp_.oc_block / 2 : tail_size < jcp_.oc_block;

            std::vector<int> kw_padding(ur_w);

            if (jcp_.exclude_pad) {
                mov(reg_tmp_32, jcp_.ic);
                imul(reg_tmp_32,  ptr[param1 + GET_OFF(kh_padding)]);

                for (int jj = 0; jj < ur_w; jj++)
                    kw_padding[jj] = 0;

                for (int ki = 0; ki < jcp_.kw; ki++) {
                    int jj_start = nstl::max(0, div_up(pad_l - ki * (jcp_.dilate_w + 1), jcp_.stride_w));
                    int jj_end = ur_w - nstl::max(0, div_up(ki * (jcp_.dilate_w + 1) + pad_r -
                                                                          (jcp_.kw - 1) * (jcp_.dilate_w + 1), jcp_.stride_w));
                    for (int jj = jj_start; jj < jj_end; jj++) {
                        kw_padding[jj]++;
                    }
                }
            } else {
                uni_vmovups(vmm_shift, ptr[reg_table + 4 * vlen]);
            }
            uni_vmovups(vmm_scale, ptr[reg_table + 3 * vlen]);

            for (int jj = 0; jj < ur_w; jj++) {
                if (jcp_.exclude_pad) {
                    mov(reg_shift, kw_padding[jj]);
                    imul(reg_shift, reg_tmp_32);
                    uni_vmovq(Xmm(vmm_shift.getIdx()), reg_shift);
                    uni_vbroadcastss(vmm_shift, Xmm(vmm_shift.getIdx()));
                    uni_vcvtdq2ps(vmm_shift, vmm_shift);
                }

                for (int ii = 0; ii < oc_blocks; ii++) {
                    uni_vcvtdq2ps(Vmm(1 + r * jcp_.ur_w * jcp_.nb_oc_blocking + ur_w * ii + jj), Vmm(1 + r * jcp_.ur_w * jcp_.nb_oc_blocking + ur_w * ii + jj));
                    uni_vfmadd213ps(Vmm(1 + r * jcp_.ur_w * jcp_.nb_oc_blocking + ur_w * ii + jj), vmm_scale, vmm_shift);
                }
            }

            int eltwise_inj_idx = 0;
            int depthwise_inj_idx = 0;
            int post_ops_data_offset = 0;
            int end_idx = jcp_.with_dw_conv ? p.find(primitive_kind::convolution) : p.len();
            for (int i = 0; i < end_idx; i++) {
                int start_idx = 1 + r * jcp_.ur_w * jcp_.nb_oc_blocking;

                auto& post_op = p.entry_[i];
                if (post_op.is_eltwise()) {
                    eltwise_injectors[eltwise_inj_idx]->compute_vector_range(start_idx, start_idx + oc_blocks * ur_w);
                    eltwise_inj_idx++;
                } else if (post_op.is_depthwise()) {
                    pop(reg_oc_off);

                    mov(reg_d_weights, ptr[reg_post_ops_data + post_ops_data_offset]);
                    add(reg_d_weights, reg_oc_off);

                    if (r == 1) {
                        add(reg_d_weights, (jcp_.oc_block / 2) * sizeof(float));
                    }

                    for (int ii = 0; ii < oc_blocks; ii++) {
                        depthwise_injectors[depthwise_inj_idx]->compute_vector_range(start_idx + ur_w * ii,
                                                                                     start_idx + ur_w * ii + ur_w, reg_d_weights, reg_d_weights);

                        add(reg_d_weights, jcp_.oc_block * sizeof(float));
                    }

                    post_ops_data_offset += depthwise_injectors[depthwise_inj_idx]->memoryStep();
                    depthwise_inj_idx++;

                    push(reg_oc_off);
                } else if (post_op.is_sum(false)) {
                    for (int ii = 0; ii < oc_blocks; ii++) {
                        for (int jj = 0; jj < ur_w; jj++) {
                            Vmm vmm_dst = Vmm(1 + r * jcp_.ur_w * jcp_.nb_oc_blocking + ur_w * ii + jj);

                            if (is_scalar_store) {
                                if (isa == x64::avx512_core) {
                                    int o_off =  jj * jcp_.oc * jcp_.ngroups;

                                    Vmm vmm_in = vmm_sum | ktail_mask | T_z;

                                    vmovups(vmm_in, ptr[reg_output + o_off * jcp_.typesize_out]);
                                    uni_vaddps(vmm_dst, vmm_dst, vmm_sum);
                                } else {
                                    for (int oc = 0; oc < tail_size; oc++) {
                                        int o_off =  jj * jcp_.oc * jcp_.ngroups + r * (jcp_.oc_block / 2) + oc;

                                        uni_vpxor(vmm_sum, vmm_sum, vmm_sum);
                                        cvt2ps(jcp_.dst_dt, vmm_sum, ptr[reg_output + o_off * jcp_.typesize_out], true);

                                        if (oc < jcp_.oc_block / 2) {
                                            uni_vpslldq(vmm_sum, vmm_sum, oc * sizeof(float));
                                        } else {
                                            Ymm ymm_prev_dst = Ymm(vmm_sum.getIdx());
                                            vperm2i128(ymm_prev_dst, ymm_prev_dst, ymm_prev_dst, 0x01);
                                            uni_vpslldq(vmm_sum, vmm_sum, (oc - jcp_.oc_block / 2) * sizeof(float));
                                        }

                                        uni_vaddps(vmm_dst, vmm_dst, vmm_sum);
                                    }
                                }
                            } else {
                                size_t o_off = ii * jcp_.oc_block + jj * jcp_.oc * jcp_.ngroups + r * (jcp_.oc_block / 2);

                                cvt2ps(jcp_.dst_dt, vmm_sum, ptr[reg_output + o_off * jcp_.typesize_out], false);
                                uni_vaddps(vmm_dst, vmm_dst, vmm_sum);
                            }
                        }
                    }
                }
            }
        }

        if (jcp_.with_binarization) {
            int binarization_idx = p.find(primitive_kind::binarization);

            OPENVINO_ASSERT(binarization_idx >= 0, "postops don't contain binarization");

            pop(reg_oc_off);

            mov(reg_b_weights, reinterpret_cast<size_t>(p.entry_[binarization_idx].binarization.weights_data));
            mov(reg_b_out_mask, reinterpret_cast<size_t>(p.entry_[binarization_idx].binarization.output_mask_data));
            add(reg_b_weights, reg_oc_off);
            add(reg_b_out_mask, reg_oc_off);

            push(reg_oc_off);

            for (int ii = 0; ii < oc_blocks; ii++) {
                for (int jj = 0; jj < ur_w; jj++) {
                    for (int r = 0; r < repeats; r++) {
                        int tail_size = isa == x64::sse41 ? nstl::min(jcp_.oc_block / 2, oc_step - r * jcp_.oc_block / 2) : oc_step;
                        mov(reg_b_mask, (1 << tail_size) - 1);
                        uni_vmovups(vmm_thr, ptr[reg_b_weights + (ii * jcp_.oc_block + r * (jcp_.oc_block / 2)) * sizeof(float)]);
                        uni_vmovups(vmm_out_mask, ptr[reg_b_out_mask + (ii * jcp_.oc_block + r * (jcp_.oc_block / 2)) * sizeof(float)]);

                        Vmm vmm_dst = Vmm(1 + r * jcp_.ur_w * jcp_.nb_oc_blocking + ur_w * ii + jj);

                        if (isa == x64::avx512_core) {
                            vcmpps(bin_mask0, vmm_dst, vmm_thr, _cmp_gt_os);
                            vptestmd(bin_mask1, vmm_out_mask, vmm_out_mask);
                            kxnorw(bin_mask0, bin_mask0, bin_mask1);
                        } else {
                            uni_vcmpgtps(vmm_dst, vmm_dst, vmm_thr);
                            uni_vpcmpeqd(vmm_dst, vmm_dst, vmm_out_mask);
                        }

                        if (r == 0) {
                            if (isa == x64::avx512_core) {
                                kmovw(reg_tmp_32, bin_mask0);
                            } else {
                                uni_vmovmskps(reg_tmp_32, vmm_dst);
                            }
                            and_(reg_tmp_64, reg_b_mask);
                        } else {
                            uni_vmovmskps(reg_tmp2_32, vmm_dst);
                            and_(reg_tmp2_64, reg_b_mask);
                            shl(reg_tmp2_32, 4);
                            or_(reg_tmp_32, reg_tmp2_32);
                        }

                        if (r == repeats - 1) {
                            if (isa == x64::avx512_core && oc_step > nbits) {
                                const size_t o_off = (2 * ii + jj * div_up(jcp_.oc, nbits));
                                mov(ptr[reg_output + o_off * jcp_.typesize_out], reg_tmp_16);
                            } else {
                                const size_t o_off = (ii + jj * div_up(jcp_.oc, nbits));
                                mov(ptr[reg_output + o_off * jcp_.typesize_out], reg_tmp_8);
                            }
                        }
                    }
                }
            }
        } else {
            for (int r = 0; r < repeats; r++) {
                int tail_size = isa == x64::sse41 ? nstl::min(jcp_.oc_block / 2, oc_step - r * jcp_.oc_block / 2) : oc_step;
                bool is_scalar_store = isa == x64::sse41 ? tail_size < jcp_.oc_block / 2 : tail_size < jcp_.oc_block;
                if (is_scalar_store) {
                    for (int jj = 0; jj < ur_w; jj++) {
                        Vmm vmm_dst = Vmm(1 + r * jcp_.ur_w * jcp_.nb_oc_blocking + jj);

                        if (isa == x64::avx512_core) {
                            size_t o_off;
                            if (jcp_.with_dw_conv)
                                o_off = jj * jcp_.oc_block;
                            else
                                o_off = jj * jcp_.oc * jcp_.ngroups;

                            uni_vmovups(ptr[reg_output + o_off * jcp_.typesize_out], vmm_dst | ktail_mask);
                        } else {
                            for (int oc = 0; oc < tail_size; oc++) {
                                size_t o_off;
                                if (jcp_.with_dw_conv)
                                    o_off = jj * jcp_.oc_block + oc + r * (jcp_.oc_block / 2);
                                else
                                    o_off = jj * jcp_.oc * jcp_.ngroups + r * (jcp_.oc_block / 2) + oc;

                                store_dst(ptr[reg_output + o_off * jcp_.typesize_out], vmm_dst, true);

                                if (isa == x64::sse41) {
                                    psrldq(vmm_dst, jcp_.typesize_out);
                                } else {
                                    Ymm ymm_dst = Ymm(1 + r * jcp_.ur_w * jcp_.nb_oc_blocking + jj);

                                    vperm2i128(ymm_tmp, ymm_dst, ymm_dst, 0x01);
                                    vpalignr(ymm_dst, vmm_tmp, ymm_dst, jcp_.typesize_out);
                                }
                            }
                        }
                    }
                } else {
                    for (int ii = 0; ii < oc_blocks; ii++) {
                        for (int jj = 0; jj < ur_w; jj++) {
                            Vmm vmm_dst = Vmm(1 + r * jcp_.ur_w * jcp_.nb_oc_blocking + ur_w * ii + jj);

                            size_t o_off;
                            if (jcp_.with_dw_conv)
                                o_off = ((size_t) ii * jcp_dw_conv_.kh * jcp_.ow + jj) * jcp_.oc_block +
                                        r * (jcp_.oc_block / 2);
                            else
                                o_off = ii * jcp_.oc_block + jj * jcp_.oc * jcp_.ngroups + r * (jcp_.oc_block / 2);

                            store_dst(ptr[reg_output + o_off * jcp_.typesize_out], vmm_dst, false);
                        }
                    }
                }
            }
        }
    }

    void solve_common(int oc_blocks, int oc_step) {
        int ur_w = jcp_.ur_w;
        int ur_w_tail = jcp_.ur_w_tail;
        int n_oi = jcp_.ow / ur_w;
        int iw = jcp_.iw;
        int kw = jcp_.kw;
        int dilate_w = jcp_.dilate_w + 1;
        int str_w = jcp_.stride_w;

        int nbits = 8;
        const int inp_mult = div_up(jcp_.ic, nbits);
        const int out_mult = jcp_.with_dw_conv ? jcp_.oc_block : jcp_.with_binarization ? div_up(jcp_.oc, nbits) : jcp_.oc;

        int l_pad = jcp_.l_pad;
        int r_pad = nstl::max(0, (jcp_.ow - 1) * str_w + (kw - 1) * dilate_w
                                 - (iw + l_pad - 1));
        int r_pad1 = (ur_w * n_oi - 1) * str_w + (kw - 1) * dilate_w
                     - (iw + l_pad - 1);
        if (r_pad1 > 0) n_oi--;

        mov(reg_input, reg_input_base);
        mov(reg_output, reg_output_base);

        push(reg_input_base);
        push(reg_output_base);
        push(reg_oc_work);
        push(reg_oc_off);

        if (l_pad > 0) {
            n_oi--;
            if (n_oi < 0 && r_pad1 > 0)
                width_blk_step(ur_w, l_pad, r_pad1, oc_blocks, oc_step); // "lrpad"
            else
                width_blk_step(ur_w, l_pad, 0, oc_blocks, oc_step); // "lpad"
            add(reg_input, jcp_.typesize_in * (ur_w * str_w - l_pad) * inp_mult);
            add(reg_output, jcp_.typesize_out * ur_w * out_mult);
        }

        Label ow_loop_label;
        xor_(oi_iter, oi_iter);

        if (n_oi > 0) {
            L(ow_loop_label);

            width_blk_step(ur_w, 0, 0, oc_blocks, oc_step); // "middle"
            add(reg_input, jcp_.typesize_in * ur_w * str_w * inp_mult);
            add(reg_output, jcp_.typesize_out * ur_w * out_mult);

            inc(oi_iter);
            cmp(oi_iter, n_oi);
            jl(ow_loop_label, T_NEAR);
        }

        if (r_pad1 > 0 && n_oi >=0) {
            width_blk_step(ur_w, 0, r_pad1, oc_blocks, oc_step); // "rpad"
            add(reg_input, jcp_.typesize_in * ur_w * str_w * inp_mult);
            add(reg_output, jcp_.typesize_out * ur_w * out_mult);
        }

        if (ur_w_tail != 0)
            width_blk_step(ur_w_tail, 0, r_pad, oc_blocks, oc_step); // "tail"

        pop(reg_oc_off);
        pop(reg_oc_work);
        pop(reg_output_base);
        pop(reg_input_base);
    }

    void prepare_table() {
        const unsigned int cvals[] = {
                0x02010100, // 0 1 1 2
                0x03020201, // 1 2 2 3
                0x03020201, // 1 2 2 3
                0x04030302,  // 2 3 3 4
                0x0f0f0f0f,
                0x000000ff,
                0xc0000000, // -2.0f
                0x01010101,
                0x00010001
        };

        size_t simd_w = vlen / sizeof(int32_t);

        align(64);
        L(l_table);
        // offset = 0
        for (size_t d = 0; d < simd_w; ++d) {
            dd(cvals[d % 4]);
        }
        // offset = 1
        for (size_t d = 0; d < simd_w; ++d) {
            dd(cvals[4]);
        }
        // offset = 2
        for (size_t d = 0; d < simd_w; ++d) {
            dd(cvals[5]);
        }
        // offset = 3
        for (size_t d = 0; d < simd_w; ++d) {
            dd(cvals[6]);
        }

        // offset = 4
        for (size_t d = 0; d < simd_w; ++d) {
            dd(x64::float2int(jcp_.ic * jcp_.kw * jcp_.kh));
        }

        // offset = 5
        for (size_t d = 0; d < simd_w; ++d) {
            dd(cvals[7]);
        }
        // offset = 6
        for (size_t d = 0; d < simd_w; ++d) {
            dd(cvals[8]);
        }
        // offset = 7
        for (size_t d = 0; d < simd_w; ++d) {
            uint32_t mask = 0xffffffff >> (jcp_.ic_padded - jcp_.ic);
            dd(mask);
        }
        // offset = 8
        for (size_t d = 0; d < simd_w; ++d) {
            uint32_t val = jcp_.pad_value == 1.0f ? 0xffffffff : 0x00000000;
            dd(val);
        }
    }
};
#endif
bool BinaryConvolution::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (isDynamicNgraphNode(op)) {
            errorMessage = "Doesn't support op with dynamic shapes";
            return false;
        }

        const auto binConv = std::dynamic_pointer_cast<const ov::opset1::BinaryConvolution>(op);
        if (!binConv) {
            errorMessage = "Only opset1 BinaryConvolution operation is supported";
            return false;
        }
        if (binConv->get_mode() != ov::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT) {
            errorMessage = "Doesn't support mode: " + ov::as_string(binConv->get_mode());
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

BinaryConvolution::BinaryConvolution(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "BinaryConvolution node with name '" + getName() + "' ";
        const auto binConv = std::dynamic_pointer_cast<const ov::opset1::BinaryConvolution>(op);

        pad_value = binConv->get_pad_value();
        for (size_t i = 0; i < binConv->get_strides().size(); i++) {
            stride.push_back(static_cast<ptrdiff_t>(binConv->get_strides()[i]));
        }
        for (size_t i = 0; i < binConv->get_dilations().size(); i++) {
            dilation.push_back(static_cast<ptrdiff_t>(binConv->get_dilations()[i]) - 1);
        }
        paddingL = binConv->get_pads_begin();
        paddingR = binConv->get_pads_end();

        if (mayiuse(x64::avx512_core)) {
            implType = impl_desc_type::jit_avx512;
        } else if (mayiuse(x64::avx2)) {
            implType = impl_desc_type::jit_avx2;
        } else if (mayiuse(x64::sse41)) {
            implType = impl_desc_type::jit_sse42;
        } else {
            implType = impl_desc_type::ref;
        }
    } else {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

void BinaryConvolution::getSupportedDescriptors() {
    withBinarization = isFusedWith(Type::FakeQuantize);
    withSum = false;
    size_t expectedInputEdgesNum = 2;
    for (size_t i = 0; i < fusedWith.size(); i++) {
        auto *eltwiseNode = dynamic_cast<Eltwise *>(fusedWith[i].get());
        if (eltwiseNode && eltwiseNode->isSpecialConvolutionAddFusing()) {
            withSum = true;
            expectedInputEdgesNum++;
        }
    }

    if (getParentEdges().size() != expectedInputEdgesNum)
        OPENVINO_THROW(errorPrefix, "has incorrect number of input edges");

    if (getChildEdges().empty())
        OPENVINO_THROW(errorPrefix, "has incorrect number of output edges");

    if (getInputShapeAtPort(0).getRank() != 4) {
        OPENVINO_THROW(errorPrefix, "doesn't support 0th input with rank: ", getInputShapeAtPort(0).getRank());
    }

    if (getInputShapeAtPort(1).getRank() != 4) {
        OPENVINO_THROW(errorPrefix, "doesn't support 1st input with rank: ", getInputShapeAtPort(1).getRank());
    }

    if (getOutputShapeAtPort(0).getRank() != 4) {
        OPENVINO_THROW(errorPrefix, "doesn't support output with rank: ", getOutputShapeAtPort(0).getRank());
    }
}

void BinaryConvolution::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    setPostOps(attr);

    NodeConfig config;
    config.inConfs.resize(2);
    config.inConfs[0].constant(false);
    config.inConfs[0].inPlace(-1);
    config.inConfs[1].constant(false);
    config.inConfs[1].inPlace(-1);

    config.outConfs.resize(1);
    config.outConfs[0].constant(false);
    config.outConfs[0].inPlace(-1);

    if (implType != impl_desc_type::ref) {
        // optimzed implementation
//        auto weiFormat = implType == impl_desc_type::jit_avx512 ? memory::format_tag::OhIw16o32i : memory::format_tag::OhIw8o32i;

        //activation
        auto nspcCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::nspc);
        config.inConfs[0].setMemDesc(nspcCreator->createSharedDesc(ov::element::u1, getInputShapeAtPort(0)));

        //weights
        size_t weiFirstDimBlockSize = implType == impl_desc_type::jit_avx512 ? 16 : 8; //memory::format_tag::OIhw16o32i : memory::format_tag::OIhw8o32i;
        auto weiDims = getInputShapeAtPort(1).getStaticDims();
        std::vector<size_t> weiBlockDims = {div_up(weiDims[0], weiFirstDimBlockSize), div_up(weiDims[1], 32),
                                            weiDims[2], weiDims[3], weiFirstDimBlockSize, 32};
        std::vector<size_t> weiOrder = {0, 1, 2, 3, 0, 1};

        config.inConfs[1].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(ov::element::u1, Shape(weiDims), weiBlockDims, weiOrder));

        //result
        auto outputPrecision = withBinarization ? ov::element::u1 : ov::element::f32;
        config.outConfs[0].setMemDesc(nspcCreator->createSharedDesc(outputPrecision, getOutputShapeAtPort(0)));
        if (withSum) {
            config.inConfs.push_back(config.outConfs[0]);
            config.outConfs[0].inPlace(2);
        }
        supportedPrimitiveDescriptors.push_back({config, implType});
    } else {
        // reference implementation
        auto weiCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
        auto nspcCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::nspc);

        config.inConfs[0].setMemDesc(nspcCreator->createSharedDesc(ov::element::u1, getInputShapeAtPort(0)));
        config.inConfs[1].setMemDesc(weiCreator->createSharedDesc(ov::element::u1, getInputShapeAtPort(1)));
        config.outConfs[0].setMemDesc(nspcCreator->createSharedDesc(ov::element::f32, getOutputShapeAtPort(0)));
        supportedPrimitiveDescriptors.push_back({config, implType});
    }
}

void BinaryConvolution::createPrimitive() {
    auto selectedPrimitiveDescriptor = getSelectedPrimitiveDescriptor();
    if (!selectedPrimitiveDescriptor)
        OPENVINO_THROW("CPU binary convolution with name '", getName(), "' doesn't have primitive descriptors.");

    auto srcDims = getParentEdgeAt(0)->getMemory().getStaticDims();
    auto weiDims = getParentEdgeAt(1)->getMemory().getStaticDims();
    auto dstDims = getChildEdgeAt(0)->getMemory().getStaticDims();

    auto implType = selectedPrimitiveDescriptor->getImplementationType();

    jcp.ngroups = group;
    jcp.mb = srcDims[0];

    jcp.oc = dstDims[1] / jcp.ngroups;
    jcp.ic = srcDims[1] / jcp.ngroups;

    jcp.ih = srcDims[2];
    jcp.iw = srcDims[3];
    jcp.oh = dstDims[2];
    jcp.ow = dstDims[3];

    bool with_groups = group > 1;
    jcp.kh = weiDims[with_groups + 2];
    jcp.kw = weiDims[with_groups + 3];

    jcp.t_pad = paddingL[0];
    jcp.b_pad = paddingR[0];
    jcp.l_pad = paddingL[1];

    jcp.stride_h = stride[0];
    jcp.stride_w = stride[1];

    jcp.dilate_h = dilation[0];
    jcp.dilate_w = dilation[1];

    jcp.pad_value = pad_value;
    jcp.exclude_pad = jcp.pad_value == 0.0f;

    jcp.with_dw_conv = false;
    jcp.with_binarization = withBinarization;

    const auto &p = (*attr.get()).post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    jcp.with_binarization = p.find(primitive_kind::binarization) != -1;

    int simd_w = implType == impl_desc_type::jit_avx512 ? 16 : 8;

    jcp.ur_w = implType == impl_desc_type::jit_avx512 ? 4 : 2;
    if (jcp.ow < jcp.ur_w) jcp.ur_w = jcp.ow;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    jcp.ic_block = 32;
    jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);
    jcp.ic_padded = rnd_up(jcp.ic, jcp.ic_block);

    jcp.oc_block = simd_w;
    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);

    jcp.nb_oc_blocking = nstl::min(implType == impl_desc_type::jit_sse42 ? 2 : implType == impl_desc_type::jit_avx2 ? 4 : 6, jcp.nb_oc);

    auto srcPrecision = getParentEdgeAt(0)->getMemory().getDesc().getPrecision();
    auto dstPrecision = getChildEdgeAt(0)->getMemory().getDesc().getPrecision();

    jcp.dst_dt = DnnlExtensionUtils::ElementTypeToDataType(dstPrecision);
    jcp.typesize_in = srcPrecision == ov::element::u1 ? 1 : srcPrecision.size();
    jcp.typesize_out = dstPrecision == ov::element::u1 ? 1 : dstPrecision.size();

    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
                                     + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));

    bool args_ok = (jcp.l_pad <= jcp.ur_w) && (r_pad_no_tail <= jcp.ur_w) &&
                   IMPLICATION(jcp.kw > 7, (jcp.t_pad == 0 && jcp.l_pad == 0) || (jcp.stride_w == 1 && jcp.stride_h == 1));
    if (!args_ok)
        OPENVINO_THROW("BinaryConvolution with name '", getName(), "' has unsupported parameters");
#if defined(OPENVINO_ARCH_X86_64)
    jit_dw_conv_params jcp_dw_conv = {};
    if (implType == impl_desc_type::jit_avx512) {
        bin_conv_kernel.reset(new jit_uni_bin_conv_kernel_f32<x64::avx512_core>(jcp, jcp_dw_conv, *attr.get()));
    } else if (implType == impl_desc_type::jit_avx2) {
        bin_conv_kernel.reset(new jit_uni_bin_conv_kernel_f32<x64::avx2>(jcp, jcp_dw_conv, *attr.get()));
    } else if (implType == impl_desc_type::sse42) {
        bin_conv_kernel.reset(new jit_uni_bin_conv_kernel_f32<x64::sse41>(jcp, jcp_dw_conv, *attr.get()));
    }
    if (bin_conv_kernel)
        bin_conv_kernel->create_ker();
#endif
}

bool BinaryConvolution::canFuse(const NodePtr& node) const {
    if (implType == impl_desc_type::ref)
        return false;

    // Binarization have to be last operation in fusing chain
    if (isFusedWith(Type::FakeQuantize))
        return false;

    if (node->getType() == Type::FakeQuantize) {
        bool ret = node->getAlgorithm() == Algorithm::FQBinarization;
        for (size_t i = 1; i < node->getParentEdges().size(); i++) {
            ret &= node->getParentEdgeAt(i)->getParent()->getChildEdges().size() == 1;
        }
        return ret;
    } else {
        return canFuseSimpleOperation(node);
    }
}

void BinaryConvolution::setPostOps(dnnl::primitive_attr &attr) {
    dnnl::post_ops ops;

    postOpsDataPtrs.clear();
    for (auto &node : fusedWith) {
        auto* eltwiseNode = dynamic_cast<Eltwise *>(node.get());
        if (eltwiseNode) {
            if (eltwiseNode->isSpecialConvolutionAddFusing()) {
                ops.append_sum(1.0);
            } else {
                // TODO [DS]: change to shape from memory
                eltwiseNode->appendPostOps(ops, getOutputShapeAtPort(0).getStaticDims(), postOpsDataPtrs);
            }
            continue;
        }

        auto* fakeQuantizeNode = dynamic_cast<FakeQuantize *>(node.get());
        if (fakeQuantizeNode) {
            fakeQuantizeNode->appendPostOps(ops, getOutputShapeAtPort(0).getStaticDims(), postOpsDataPtrs);
            continue;
        }

        OPENVINO_THROW("Fusing of ",
                       NameFromType(node->getType()),
                       " operation to ",
                       NameFromType(this->getType()),
                       " node is not implemented");
    }

    attr.set_post_ops(ops);
}

void BinaryConvolution::executeOptimized(const uint8_t* src, const uint8_t* weights, uint8_t* dst,
                                         const std::vector<size_t>& s_str, const std::vector<size_t>& w_str, const std::vector<size_t>& d_str) {
    auto dst_f32 = reinterpret_cast<float *>(dst);

    const int MB = jcp.mb;

    int ocb_work = div_up(jcp.nb_oc, jcp.nb_oc_blocking);
    int nbits = 8;

    parallel_for4d(MB, jcp.ngroups, ocb_work, jcp.oh, [&](int n, int g, int ocbb, int oh) {
        int ocb = ocbb * jcp.nb_oc_blocking;
        int ocb_num = jcp.nb_oc_blocking;

        auto par_conv = jit_bin_conv_call_args();

        const int ij = oh * jcp.stride_h;
        const int i_t_overflow = nstl::min(jcp.kh, div_up(nstl::max(0, jcp.t_pad - ij), (jcp.dilate_h+1)));
        const int i_b_overflow = nstl::min(jcp.kh, div_up(nstl::max(jcp.ih, ij + (jcp.kh-1) * (jcp.dilate_h+1) -
                                                                                          jcp.t_pad+1) - jcp.ih, (jcp.dilate_h + 1)));

        const size_t _oc = g * jcp.nb_oc + ocb;
        const size_t _ic = g * jcp.nb_ic;

        const int ih = nstl::max(ij - jcp.t_pad + i_t_overflow * (jcp.dilate_h + 1), 0);
        par_conv.src = &src[(n * s_str[0] + _ic*jcp.ic_block * s_str[1] + ih * s_str[2]) / nbits];

        if (jcp.with_binarization) {
            par_conv.dst = &dst[(n * d_str[0] + _oc*jcp.oc_block * d_str[1] + oh * d_str[2]) / nbits];
        } else {
            par_conv.dst = &dst_f32[n * d_str[0] + _oc*jcp.oc_block * d_str[1] + oh * d_str[2]];
        }

        const int wh = jcp.exclude_pad ? i_t_overflow : 0;
        par_conv.filt = &weights[(ocb * w_str[0] + wh * w_str[2]) / nbits];

        par_conv.oc_work = nstl::min((ocb + ocb_num) * jcp.oc_block, jcp.oc) - ocb*jcp.oc_block;

        par_conv.kw_padding = 0;
        const int kh_padding = jcp.kh - i_t_overflow - i_b_overflow;
        par_conv.kh_padding = nstl::max(0, kh_padding);
        par_conv.t_overflow = i_t_overflow;
        par_conv.b_overflow = i_b_overflow;

        par_conv.oc_off = _oc * jcp.oc_block * sizeof(float);
        par_conv.post_op_data = postOpsDataPtrs.data();

        (*bin_conv_kernel)(&par_conv);
    });
}

void BinaryConvolution::executeReference(const uint8_t* src, const uint8_t* weights, uint8_t* dst,
                                                   const std::vector<size_t>& s_str, const std::vector<size_t>& w_str, const std::vector<size_t>& d_str) {
    auto dst_fp = reinterpret_cast<float *>(dst);

    const bool with_groups = jcp.ngroups > 1;

    const int G = jcp.ngroups;
    const int MB = jcp.mb;
    const int OH = jcp.oh;
    const int OW = jcp.ow;
    const int IH = jcp.ih;
    const int IW = jcp.iw;

    const int OC = jcp.oc;
    const int IC = jcp.ic;

    const int KH = jcp.kh;
    const int KW = jcp.kw;

    const int KSH = jcp.stride_h;
    const int KSW = jcp.stride_w;

    const int KDH = jcp.dilate_h;
    const int KDW = jcp.dilate_w;

    const int padT = jcp.t_pad;
    const int padL = jcp.l_pad;

    const float pad_value = jcp.pad_value;

    const int nbits = 8;

    auto extract_bit = [](uint8_t val, uint8_t bit) -> uint8_t {
        return (uint8_t)((val >> bit) & 0x0001);
    };

    auto ker = [=](int32_t &d, int g, int mb, int oc, int oh, int ow) {
        for (int ic = 0; ic < IC; ++ic) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    const int ih = oh * KSH - padT + kh * (1 + KDH);
                    const int iw = ow * KSW - padL + kw * (1 + KDW);

                    size_t iidx = 0;
                    size_t widx = 0;

                    iidx = mb * s_str[0] + (g * IC + ic) * s_str[1] + ih * s_str[2] + iw * s_str[3];
                    widx = with_groups ? g * w_str[0] + oc * w_str[1] + ic * w_str[2] + kh * w_str[3] + kw * w_str[4]
                                       : oc * w_str[0] + ic * w_str[1] + kh * w_str[2] + kw * w_str[3];

                    uint8_t s;
                    if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) {
                        if (pad_value == 0)
                            continue;
                        else
                            s = pad_value == 1.0f ? (uint8_t) 1 : (uint8_t) 0;
                    } else {
                        s = extract_bit(src[iidx / nbits], (uint8_t) (iidx % nbits));
                    }

                    uint8_t w = extract_bit(weights[widx / nbits], (uint8_t) (widx % nbits));

                    d += (int32_t) (s ^ w);
                }
            }
        }
    };

    parallel_for5d(G, MB, OC, OH, OW, [&](int g, int mb, int oc, int oh, int ow) {
        int32_t a = 0;
        ker(a, g, mb, oc, oh, ow);

        float base_value;
        if (pad_value == 0.0f) {
            const int i_left_overflow = nstl::max(0, (padL - ow * KSW));
            const int i_right_overflow = nstl::max(IW, (ow * KSW + (KW - 1) * (KDW + 1) - padL + 1)) - IW;
            const int kw_padding =
                    KW - div_up(i_left_overflow, (KDW + 1)) - div_up(i_right_overflow, (KDW + 1));

            const int i_top_overflow = nstl::max(0, (padT - oh * KSH));
            const int i_bottom_overflow = nstl::max(IH, (oh * KSH + (KH - 1) * (KDH + 1) - padT + 1)) - IH;
            const int kh_padding =
                    KH - div_up(i_top_overflow, (KDH + 1)) - div_up(i_bottom_overflow, (KDH + 1));

            base_value = IC * kh_padding * kw_padding;
        } else {
            base_value = IC * KH * KW;
        }

        float a_fp = base_value - static_cast<float>(2 * a);

        dst_fp[mb * d_str[0] + (g*OC + oc) * d_str[1] + oh * d_str[2] + ow * d_str[3]] = a_fp;
    });
}

void BinaryConvolution::execute(dnnl::stream strm) {
    auto srcMemory = getSrcMemoryAtPort(0);
    auto weightsMemory = getSrcMemoryAtPort(1);
    auto dstMemory = getDstMemoryAtPort(0);

    auto src = srcMemory->getDataAs<const uint8_t>();
    auto weights = weightsMemory->getDataAs<const uint8_t>();
    auto dst = dstMemory->getDataAs<uint8_t>();

    auto srcDesc = getParentEdgeAt(0)->getMemory().getDescWithType<BlockedMemoryDesc>();
    std::vector<size_t> srcStride(srcDesc->getStrides().size());
    for (size_t i = 0; i < srcStride.size(); i++) {
        srcStride[srcDesc->getOrder()[i]] = srcDesc->getStrides()[i];
    }

    auto weiDesc = getParentEdgeAt(1)->getMemory().getDescWithType<BlockedMemoryDesc>();
    std::vector<size_t> weightsStride(weiDesc->getShape().getRank());
    for (size_t i = 0; i < weightsStride.size(); i++) {
        weightsStride[weiDesc->getOrder()[i]] = weiDesc->getStrides()[i];
    }

    auto dstDesc = getChildEdgeAt(0)->getMemory().getDescWithType<BlockedMemoryDesc>();
    std::vector<size_t> dstStride(dstDesc->getStrides().size());
    for (size_t i = 0; i < dstStride.size(); i++) {
        dstStride[dstDesc->getOrder()[i]] = dstDesc->getStrides()[i];
    }

    auto selectedPrimitiveDescriptor = getSelectedPrimitiveDescriptor();
    if (!selectedPrimitiveDescriptor)
        OPENVINO_THROW("CPU binary convolution with name '", getName(), "' doesn't have primitive descriptors.");

    auto implType = selectedPrimitiveDescriptor->getImplementationType();
    if (implType != impl_desc_type::ref) {
        executeOptimized(src, weights, dst, srcStride, weightsStride, dstStride);
    } else {
        executeReference(src, weights, dst, srcStride, weightsStride, dstStride);
    }
}

bool BinaryConvolution::created() const {
    return getType() == Type::BinaryConvolution;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
