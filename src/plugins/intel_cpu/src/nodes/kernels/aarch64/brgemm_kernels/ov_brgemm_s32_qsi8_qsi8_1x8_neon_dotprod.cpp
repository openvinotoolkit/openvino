// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/aarch64/brgemm_kernels/int8_brgemm_kernels.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_gen.h>

#include "utils/cpu_utils.hpp"

namespace ov::intel_cpu::aarch64 {

using namespace Xbyak_aarch64;
using namespace dnnl::impl::cpu::aarch64;

jit_int8_brgemm_kernel_1x8_dot::jit_int8_brgemm_kernel_1x8_dot() : jit_generator() {}

void jit_int8_brgemm_kernel_1x8_dot::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_1x8_dot::generate() {
    preamble();

    const XReg reg_src = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_ldb = abi_param5;
    const XReg reg_accum = abi_param6;

    const XReg reg_w0 = x10;
    const XReg reg_w1 = x11;
    const XReg reg_w2 = x12;
    const XReg reg_w3 = x13;
    const XReg reg_w4 = x14;
    const XReg reg_w5 = x15;
    const XReg reg_w6 = x16;
    const XReg reg_w7 = x17;

    const WReg reg_a = w26;
    const WReg reg_b0 = w18;
    const WReg reg_b1 = w19;
    const WReg reg_b2 = w20;
    const WReg reg_b3 = w21;
    const WReg reg_b4 = w22;
    const WReg reg_b5 = w23;
    const WReg reg_b6 = w24;
    const WReg reg_b7 = w25;

    const WReg reg_acc0 = w4;
    const WReg reg_acc1 = w5;
    const WReg reg_acc2 = w6;
    const WReg reg_acc3 = w7;
    const WReg reg_acc4 = w8;
    const WReg reg_acc5 = w9;
    // Keep tail accumulators disjoint from x10/x11 weight pointers.
    const WReg reg_acc6 = w27;
    const WReg reg_acc7 = w28;

    const VReg16B v_src = VReg16B(0);
    const VReg16B v_w0 = VReg16B(1);
    const VReg16B v_w1 = VReg16B(2);
    const VReg16B v_w2 = VReg16B(3);
    const VReg16B v_w3 = VReg16B(4);
    const VReg16B v_w4 = VReg16B(5);
    const VReg16B v_w5 = VReg16B(6);
    const VReg16B v_w6 = VReg16B(7);
    const VReg16B v_w7 = VReg16B(8);

    const VReg4S v_acc0 = VReg4S(16);
    const VReg4S v_acc1 = VReg4S(17);
    const VReg4S v_acc2 = VReg4S(18);
    const VReg4S v_acc3 = VReg4S(19);
    const VReg4S v_acc4 = VReg4S(20);
    const VReg4S v_acc5 = VReg4S(21);
    const VReg4S v_acc6 = VReg4S(22);
    const VReg4S v_acc7 = VReg4S(23);

    const VReg16B v_acc0b(v_acc0.getIdx());
    const VReg16B v_acc1b(v_acc1.getIdx());
    const VReg16B v_acc2b(v_acc2.getIdx());
    const VReg16B v_acc3b(v_acc3.getIdx());
    const VReg16B v_acc4b(v_acc4.getIdx());
    const VReg16B v_acc5b(v_acc5.getIdx());
    const VReg16B v_acc6b(v_acc6.getIdx());
    const VReg16B v_acc7b(v_acc7.getIdx());

    eor(v_acc0b, v_acc0b, v_acc0b);
    eor(v_acc1b, v_acc1b, v_acc1b);
    eor(v_acc2b, v_acc2b, v_acc2b);
    eor(v_acc3b, v_acc3b, v_acc3b);
    eor(v_acc4b, v_acc4b, v_acc4b);
    eor(v_acc5b, v_acc5b, v_acc5b);
    eor(v_acc6b, v_acc6b, v_acc6b);
    eor(v_acc7b, v_acc7b, v_acc7b);

    mov(reg_w0, reg_wei);
    add(reg_w1, reg_w0, reg_ldb);
    add(reg_w2, reg_w1, reg_ldb);
    add(reg_w3, reg_w2, reg_ldb);
    add(reg_w4, reg_w3, reg_ldb);
    add(reg_w5, reg_w4, reg_ldb);
    add(reg_w6, reg_w5, reg_ldb);
    add(reg_w7, reg_w6, reg_ldb);

    Label loop32;
    Label loop16;
    Label tail;
    Label tail_loop;
    Label done;
    Label store;

    cmp(reg_k, 32);
    b(LT, loop16);

    L(loop32);
    ld1(v_src, ptr(reg_src));
    ld1(v_w0, ptr(reg_w0));
    ld1(v_w1, ptr(reg_w1));
    ld1(v_w2, ptr(reg_w2));
    ld1(v_w3, ptr(reg_w3));
    ld1(v_w4, ptr(reg_w4));
    ld1(v_w5, ptr(reg_w5));
    ld1(v_w6, ptr(reg_w6));
    ld1(v_w7, ptr(reg_w7));

    add(reg_src, reg_src, 16);
    add(reg_w0, reg_w0, 16);
    add(reg_w1, reg_w1, 16);
    add(reg_w2, reg_w2, 16);
    add(reg_w3, reg_w3, 16);
    add(reg_w4, reg_w4, 16);
    add(reg_w5, reg_w5, 16);
    add(reg_w6, reg_w6, 16);
    add(reg_w7, reg_w7, 16);

    sdot(v_acc0, v_src, v_w0);
    sdot(v_acc1, v_src, v_w1);
    sdot(v_acc2, v_src, v_w2);
    sdot(v_acc3, v_src, v_w3);
    sdot(v_acc4, v_src, v_w4);
    sdot(v_acc5, v_src, v_w5);
    sdot(v_acc6, v_src, v_w6);
    sdot(v_acc7, v_src, v_w7);

    ld1(v_src, ptr(reg_src));
    ld1(v_w0, ptr(reg_w0));
    ld1(v_w1, ptr(reg_w1));
    ld1(v_w2, ptr(reg_w2));
    ld1(v_w3, ptr(reg_w3));
    ld1(v_w4, ptr(reg_w4));
    ld1(v_w5, ptr(reg_w5));
    ld1(v_w6, ptr(reg_w6));
    ld1(v_w7, ptr(reg_w7));

    add(reg_src, reg_src, 16);
    add(reg_w0, reg_w0, 16);
    add(reg_w1, reg_w1, 16);
    add(reg_w2, reg_w2, 16);
    add(reg_w3, reg_w3, 16);
    add(reg_w4, reg_w4, 16);
    add(reg_w5, reg_w5, 16);
    add(reg_w6, reg_w6, 16);
    add(reg_w7, reg_w7, 16);

    sdot(v_acc0, v_src, v_w0);
    sdot(v_acc1, v_src, v_w1);
    sdot(v_acc2, v_src, v_w2);
    sdot(v_acc3, v_src, v_w3);
    sdot(v_acc4, v_src, v_w4);
    sdot(v_acc5, v_src, v_w5);
    sdot(v_acc6, v_src, v_w6);
    sdot(v_acc7, v_src, v_w7);

    sub(reg_k, reg_k, 32);
    cmp(reg_k, 32);
    b(GE, loop32);

    L(loop16);
    cmp(reg_k, 16);
    b(LT, tail);
    ld1(v_src, ptr(reg_src));
    ld1(v_w0, ptr(reg_w0));
    ld1(v_w1, ptr(reg_w1));
    ld1(v_w2, ptr(reg_w2));
    ld1(v_w3, ptr(reg_w3));
    ld1(v_w4, ptr(reg_w4));
    ld1(v_w5, ptr(reg_w5));
    ld1(v_w6, ptr(reg_w6));
    ld1(v_w7, ptr(reg_w7));

    add(reg_src, reg_src, 16);
    add(reg_w0, reg_w0, 16);
    add(reg_w1, reg_w1, 16);
    add(reg_w2, reg_w2, 16);
    add(reg_w3, reg_w3, 16);
    add(reg_w4, reg_w4, 16);
    add(reg_w5, reg_w5, 16);
    add(reg_w6, reg_w6, 16);
    add(reg_w7, reg_w7, 16);

    sdot(v_acc0, v_src, v_w0);
    sdot(v_acc1, v_src, v_w1);
    sdot(v_acc2, v_src, v_w2);
    sdot(v_acc3, v_src, v_w3);
    sdot(v_acc4, v_src, v_w4);
    sdot(v_acc5, v_src, v_w5);
    sdot(v_acc6, v_src, v_w6);
    sdot(v_acc7, v_src, v_w7);

    sub(reg_k, reg_k, 16);

    L(tail);
    const VReg v_tmp(24);
    const SReg s_tmp(v_tmp.getIdx());

    addv(s_tmp, v_acc0);
    umov(reg_acc0, v_tmp.s[0]);
    addv(s_tmp, v_acc1);
    umov(reg_acc1, v_tmp.s[0]);
    addv(s_tmp, v_acc2);
    umov(reg_acc2, v_tmp.s[0]);
    addv(s_tmp, v_acc3);
    umov(reg_acc3, v_tmp.s[0]);
    addv(s_tmp, v_acc4);
    umov(reg_acc4, v_tmp.s[0]);
    addv(s_tmp, v_acc5);
    umov(reg_acc5, v_tmp.s[0]);
    addv(s_tmp, v_acc6);
    umov(reg_acc6, v_tmp.s[0]);
    addv(s_tmp, v_acc7);
    umov(reg_acc7, v_tmp.s[0]);

    cbz(reg_k, done);

    L(tail_loop);
    ldrsb(reg_a, ptr(reg_src));
    add(reg_src, reg_src, 1);
    ldrsb(reg_b0, ptr(reg_w0));
    ldrsb(reg_b1, ptr(reg_w1));
    ldrsb(reg_b2, ptr(reg_w2));
    ldrsb(reg_b3, ptr(reg_w3));
    ldrsb(reg_b4, ptr(reg_w4));
    ldrsb(reg_b5, ptr(reg_w5));
    ldrsb(reg_b6, ptr(reg_w6));
    ldrsb(reg_b7, ptr(reg_w7));

    add(reg_w0, reg_w0, 1);
    add(reg_w1, reg_w1, 1);
    add(reg_w2, reg_w2, 1);
    add(reg_w3, reg_w3, 1);
    add(reg_w4, reg_w4, 1);
    add(reg_w5, reg_w5, 1);
    add(reg_w6, reg_w6, 1);
    add(reg_w7, reg_w7, 1);

    madd(reg_acc0, reg_a, reg_b0, reg_acc0);
    madd(reg_acc1, reg_a, reg_b1, reg_acc1);
    madd(reg_acc2, reg_a, reg_b2, reg_acc2);
    madd(reg_acc3, reg_a, reg_b3, reg_acc3);
    madd(reg_acc4, reg_a, reg_b4, reg_acc4);
    madd(reg_acc5, reg_a, reg_b5, reg_acc5);
    madd(reg_acc6, reg_a, reg_b6, reg_acc6);
    madd(reg_acc7, reg_a, reg_b7, reg_acc7);

    subs(reg_k, reg_k, 1);
    b(NE, tail_loop);

    L(done);
    cbz(reg_accum, store);
    const WReg reg_tmp = w12;
    ldr(reg_tmp, ptr(reg_dst));
    add(reg_acc0, reg_acc0, reg_tmp);
    ldr(reg_tmp, ptr(reg_dst, 4));
    add(reg_acc1, reg_acc1, reg_tmp);
    ldr(reg_tmp, ptr(reg_dst, 8));
    add(reg_acc2, reg_acc2, reg_tmp);
    ldr(reg_tmp, ptr(reg_dst, 12));
    add(reg_acc3, reg_acc3, reg_tmp);
    ldr(reg_tmp, ptr(reg_dst, 16));
    add(reg_acc4, reg_acc4, reg_tmp);
    ldr(reg_tmp, ptr(reg_dst, 20));
    add(reg_acc5, reg_acc5, reg_tmp);
    ldr(reg_tmp, ptr(reg_dst, 24));
    add(reg_acc6, reg_acc6, reg_tmp);
    ldr(reg_tmp, ptr(reg_dst, 28));
    add(reg_acc7, reg_acc7, reg_tmp);

    L(store);
    str(reg_acc0, ptr(reg_dst));
    str(reg_acc1, ptr(reg_dst, 4));
    str(reg_acc2, ptr(reg_dst, 8));
    str(reg_acc3, ptr(reg_dst, 12));
    str(reg_acc4, ptr(reg_dst, 16));
    str(reg_acc5, ptr(reg_dst, 20));
    str(reg_acc6, ptr(reg_dst, 24));
    str(reg_acc7, ptr(reg_dst, 28));

    postamble();
}

}  // namespace ov::intel_cpu::aarch64
