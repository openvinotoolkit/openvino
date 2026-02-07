// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_int8_conv_kernel.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_gen.h>

#include "utils/cpu_utils.hpp"

namespace ov::intel_cpu::aarch64 {

using namespace Xbyak_aarch64;
using namespace dnnl::impl::cpu::aarch64;

jit_int8_dot_kernel::jit_int8_dot_kernel(bool src_signed) : jit_generator(), src_signed_(src_signed) {}

void jit_int8_dot_kernel::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_dot_kernel::generate() {
    preamble();

    const XReg reg_src = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_accum = abi_param5;

    const WReg reg_acc = w4;
    const WReg reg_a = w5;
    const WReg reg_b = w6;

    Label init_zero;
    Label init_done;

    cbz(reg_accum, init_zero);
    ldr(reg_acc, ptr(reg_dst));
    b(init_done);

    L(init_zero);
    mov(reg_acc, 0);
    L(init_done);

    Label loop;
    Label done;

    cmp(reg_k, 0);
    b(EQ, done);

    L(loop);
    if (src_signed_) {
        ldrsb(reg_a, ptr(reg_src));
    } else {
        ldrb(reg_a, ptr(reg_src));
    }
    add(reg_src, reg_src, 1);
    ldrsb(reg_b, ptr(reg_wei));
    add(reg_wei, reg_wei, 1);
    madd(reg_acc, reg_a, reg_b, reg_acc);
    subs(reg_k, reg_k, 1);
    b(NE, loop);

    L(done);
    str(reg_acc, ptr(reg_dst));

    postamble();
}

jit_int8_brgemm_kernel_1x4::jit_int8_brgemm_kernel_1x4(bool src_signed) : jit_generator(), src_signed_(src_signed) {}

void jit_int8_brgemm_kernel_1x4::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_1x4::generate() {
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

    const WReg reg_a = w8;
    const WReg reg_b0 = w14;
    const WReg reg_b1 = w15;
    const WReg reg_b2 = w16;
    const WReg reg_b3 = w17;

    const WReg reg_acc0 = w4;
    const WReg reg_acc1 = w5;
    const WReg reg_acc2 = w6;
    const WReg reg_acc3 = w7;

    Label init_zero;
    Label init_done;

    cbz(reg_accum, init_zero);
    ldr(reg_acc0, ptr(reg_dst));
    ldr(reg_acc1, ptr(reg_dst, 4));
    ldr(reg_acc2, ptr(reg_dst, 8));
    ldr(reg_acc3, ptr(reg_dst, 12));
    b(init_done);

    L(init_zero);
    mov(reg_acc0, 0);
    mov(reg_acc1, 0);
    mov(reg_acc2, 0);
    mov(reg_acc3, 0);
    L(init_done);

    mov(reg_w0, reg_wei);
    add(reg_w1, reg_w0, reg_ldb);
    add(reg_w2, reg_w1, reg_ldb);
    add(reg_w3, reg_w2, reg_ldb);

    Label loop;
    Label done;

    cmp(reg_k, 0);
    b(EQ, done);

    L(loop);
    if (src_signed_) {
        ldrsb(reg_a, ptr(reg_src));
    } else {
        ldrb(reg_a, ptr(reg_src));
    }
    add(reg_src, reg_src, 1);

    ldrsb(reg_b0, ptr(reg_w0));
    ldrsb(reg_b1, ptr(reg_w1));
    ldrsb(reg_b2, ptr(reg_w2));
    ldrsb(reg_b3, ptr(reg_w3));

    add(reg_w0, reg_w0, 1);
    add(reg_w1, reg_w1, 1);
    add(reg_w2, reg_w2, 1);
    add(reg_w3, reg_w3, 1);

    madd(reg_acc0, reg_a, reg_b0, reg_acc0);
    madd(reg_acc1, reg_a, reg_b1, reg_acc1);
    madd(reg_acc2, reg_a, reg_b2, reg_acc2);
    madd(reg_acc3, reg_a, reg_b3, reg_acc3);

    subs(reg_k, reg_k, 1);
    b(NE, loop);

    L(done);
    str(reg_acc0, ptr(reg_dst));
    str(reg_acc1, ptr(reg_dst, 4));
    str(reg_acc2, ptr(reg_dst, 8));
    str(reg_acc3, ptr(reg_dst, 12));

    postamble();
}

jit_int8_brgemm_kernel_1x4_dot::jit_int8_brgemm_kernel_1x4_dot() : jit_generator() {}

void jit_int8_brgemm_kernel_1x4_dot::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_1x4_dot::generate() {
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

    const WReg reg_a = w8;
    const WReg reg_b0 = w14;
    const WReg reg_b1 = w15;
    const WReg reg_b2 = w16;
    const WReg reg_b3 = w17;

    const WReg reg_acc0 = w4;
    const WReg reg_acc1 = w5;
    const WReg reg_acc2 = w6;
    const WReg reg_acc3 = w7;

    const VReg16B v_src = VReg16B(0);
    const VReg16B v_w0 = VReg16B(1);
    const VReg16B v_w1 = VReg16B(2);
    const VReg16B v_w2 = VReg16B(3);
    const VReg16B v_w3 = VReg16B(4);

    const VReg4S v_acc0 = VReg4S(20);
    const VReg4S v_acc1 = VReg4S(21);
    const VReg4S v_acc2 = VReg4S(22);
    const VReg4S v_acc3 = VReg4S(23);

    const VReg v_tmp0 = VReg(24);
    const VReg v_tmp1 = VReg(25);
    const VReg v_tmp2 = VReg(26);
    const VReg v_tmp3 = VReg(27);

    const VReg16B v_acc0b(v_acc0.getIdx());
    const VReg16B v_acc1b(v_acc1.getIdx());
    const VReg16B v_acc2b(v_acc2.getIdx());
    const VReg16B v_acc3b(v_acc3.getIdx());

    eor(v_acc0b, v_acc0b, v_acc0b);
    eor(v_acc1b, v_acc1b, v_acc1b);
    eor(v_acc2b, v_acc2b, v_acc2b);
    eor(v_acc3b, v_acc3b, v_acc3b);

    mov(reg_w0, reg_wei);
    add(reg_w1, reg_w0, reg_ldb);
    add(reg_w2, reg_w1, reg_ldb);
    add(reg_w3, reg_w2, reg_ldb);

    Label loop;
    Label tail;
    Label tail_loop;
    Label done;
    Label store;

    cmp(reg_k, 16);
    b(LT, tail);

    L(loop);
    ld1(v_src, ptr(reg_src));
    ld1(v_w0, ptr(reg_w0));
    ld1(v_w1, ptr(reg_w1));
    ld1(v_w2, ptr(reg_w2));
    ld1(v_w3, ptr(reg_w3));

    add(reg_src, reg_src, 16);
    add(reg_w0, reg_w0, 16);
    add(reg_w1, reg_w1, 16);
    add(reg_w2, reg_w2, 16);
    add(reg_w3, reg_w3, 16);

    sdot(v_acc0, v_src, v_w0);
    sdot(v_acc1, v_src, v_w1);
    sdot(v_acc2, v_src, v_w2);
    sdot(v_acc3, v_src, v_w3);

    sub(reg_k, reg_k, 16);
    cmp(reg_k, 16);
    b(GE, loop);

    L(tail);
    const SReg s_tmp0(v_tmp0.getIdx());
    const SReg s_tmp1(v_tmp1.getIdx());
    const SReg s_tmp2(v_tmp2.getIdx());
    const SReg s_tmp3(v_tmp3.getIdx());

    addv(s_tmp0, v_acc0);
    addv(s_tmp1, v_acc1);
    addv(s_tmp2, v_acc2);
    addv(s_tmp3, v_acc3);

    umov(reg_acc0, v_tmp0.s[0]);
    umov(reg_acc1, v_tmp1.s[0]);
    umov(reg_acc2, v_tmp2.s[0]);
    umov(reg_acc3, v_tmp3.s[0]);

    cbz(reg_k, done);

    L(tail_loop);
    ldrsb(reg_a, ptr(reg_src));
    add(reg_src, reg_src, 1);
    ldrsb(reg_b0, ptr(reg_w0));
    ldrsb(reg_b1, ptr(reg_w1));
    ldrsb(reg_b2, ptr(reg_w2));
    ldrsb(reg_b3, ptr(reg_w3));
    add(reg_w0, reg_w0, 1);
    add(reg_w1, reg_w1, 1);
    add(reg_w2, reg_w2, 1);
    add(reg_w3, reg_w3, 1);
    madd(reg_acc0, reg_a, reg_b0, reg_acc0);
    madd(reg_acc1, reg_a, reg_b1, reg_acc1);
    madd(reg_acc2, reg_a, reg_b2, reg_acc2);
    madd(reg_acc3, reg_a, reg_b3, reg_acc3);
    subs(reg_k, reg_k, 1);
    b(NE, tail_loop);

    L(done);
    cbz(reg_accum, store);
    const WReg reg_tmp = w9;
    ldr(reg_tmp, ptr(reg_dst));
    add(reg_acc0, reg_acc0, reg_tmp);
    ldr(reg_tmp, ptr(reg_dst, 4));
    add(reg_acc1, reg_acc1, reg_tmp);
    ldr(reg_tmp, ptr(reg_dst, 8));
    add(reg_acc2, reg_acc2, reg_tmp);
    ldr(reg_tmp, ptr(reg_dst, 12));
    add(reg_acc3, reg_acc3, reg_tmp);

    L(store);
    str(reg_acc0, ptr(reg_dst));
    str(reg_acc1, ptr(reg_dst, 4));
    str(reg_acc2, ptr(reg_dst, 8));
    str(reg_acc3, ptr(reg_dst, 12));

    postamble();
}

jit_int8_brgemm_kernel_1x4_udot::jit_int8_brgemm_kernel_1x4_udot() : jit_generator() {}

void jit_int8_brgemm_kernel_1x4_udot::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_1x4_udot::generate() {
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

    const WReg reg_a = w8;
    const WReg reg_b0 = w14;
    const WReg reg_b1 = w15;
    const WReg reg_b2 = w16;
    const WReg reg_b3 = w17;

    const WReg reg_acc0 = w4;
    const WReg reg_acc1 = w5;
    const WReg reg_acc2 = w6;
    const WReg reg_acc3 = w7;
    const WReg reg_mask = w9;

    const VReg16B v_src = VReg16B(0);
    const VReg16B v_w0 = VReg16B(1);
    const VReg16B v_w1 = VReg16B(2);
    const VReg16B v_w2 = VReg16B(3);
    const VReg16B v_w3 = VReg16B(4);
    const VReg16B v_mask = VReg16B(5);

    const VReg4S v_acc0 = VReg4S(20);
    const VReg4S v_acc1 = VReg4S(21);
    const VReg4S v_acc2 = VReg4S(22);
    const VReg4S v_acc3 = VReg4S(23);

    const VReg v_tmp0 = VReg(24);
    const VReg v_tmp1 = VReg(25);
    const VReg v_tmp2 = VReg(26);
    const VReg v_tmp3 = VReg(27);

    const VReg16B v_acc0b(v_acc0.getIdx());
    const VReg16B v_acc1b(v_acc1.getIdx());
    const VReg16B v_acc2b(v_acc2.getIdx());
    const VReg16B v_acc3b(v_acc3.getIdx());

    mov(reg_mask, 0x80);
    dup(v_mask, reg_mask);

    eor(v_acc0b, v_acc0b, v_acc0b);
    eor(v_acc1b, v_acc1b, v_acc1b);
    eor(v_acc2b, v_acc2b, v_acc2b);
    eor(v_acc3b, v_acc3b, v_acc3b);

    mov(reg_w0, reg_wei);
    add(reg_w1, reg_w0, reg_ldb);
    add(reg_w2, reg_w1, reg_ldb);
    add(reg_w3, reg_w2, reg_ldb);

    Label loop;
    Label tail;
    Label tail_loop;
    Label done;
    Label store;

    cmp(reg_k, 16);
    b(LT, tail);

    L(loop);
    ld1(v_src, ptr(reg_src));
    ld1(v_w0, ptr(reg_w0));
    ld1(v_w1, ptr(reg_w1));
    ld1(v_w2, ptr(reg_w2));
    ld1(v_w3, ptr(reg_w3));

    eor(v_src, v_src, v_mask);

    add(reg_src, reg_src, 16);
    add(reg_w0, reg_w0, 16);
    add(reg_w1, reg_w1, 16);
    add(reg_w2, reg_w2, 16);
    add(reg_w3, reg_w3, 16);

    sdot(v_acc0, v_src, v_w0);
    sdot(v_acc1, v_src, v_w1);
    sdot(v_acc2, v_src, v_w2);
    sdot(v_acc3, v_src, v_w3);

    sub(reg_k, reg_k, 16);
    cmp(reg_k, 16);
    b(GE, loop);

    L(tail);
    const SReg s_tmp0(v_tmp0.getIdx());
    const SReg s_tmp1(v_tmp1.getIdx());
    const SReg s_tmp2(v_tmp2.getIdx());
    const SReg s_tmp3(v_tmp3.getIdx());

    addv(s_tmp0, v_acc0);
    addv(s_tmp1, v_acc1);
    addv(s_tmp2, v_acc2);
    addv(s_tmp3, v_acc3);

    umov(reg_acc0, v_tmp0.s[0]);
    umov(reg_acc1, v_tmp1.s[0]);
    umov(reg_acc2, v_tmp2.s[0]);
    umov(reg_acc3, v_tmp3.s[0]);

    cbz(reg_k, done);

    L(tail_loop);
    ldrb(reg_a, ptr(reg_src));
    sub(reg_a, reg_a, 128);
    add(reg_src, reg_src, 1);
    ldrsb(reg_b0, ptr(reg_w0));
    ldrsb(reg_b1, ptr(reg_w1));
    ldrsb(reg_b2, ptr(reg_w2));
    ldrsb(reg_b3, ptr(reg_w3));
    add(reg_w0, reg_w0, 1);
    add(reg_w1, reg_w1, 1);
    add(reg_w2, reg_w2, 1);
    add(reg_w3, reg_w3, 1);
    madd(reg_acc0, reg_a, reg_b0, reg_acc0);
    madd(reg_acc1, reg_a, reg_b1, reg_acc1);
    madd(reg_acc2, reg_a, reg_b2, reg_acc2);
    madd(reg_acc3, reg_a, reg_b3, reg_acc3);
    subs(reg_k, reg_k, 1);
    b(NE, tail_loop);

    L(done);
    cbz(reg_accum, store);
    const WReg reg_tmp = w9;
    ldr(reg_tmp, ptr(reg_dst));
    add(reg_acc0, reg_acc0, reg_tmp);
    ldr(reg_tmp, ptr(reg_dst, 4));
    add(reg_acc1, reg_acc1, reg_tmp);
    ldr(reg_tmp, ptr(reg_dst, 8));
    add(reg_acc2, reg_acc2, reg_tmp);
    ldr(reg_tmp, ptr(reg_dst, 12));
    add(reg_acc3, reg_acc3, reg_tmp);

    L(store);
    str(reg_acc0, ptr(reg_dst));
    str(reg_acc1, ptr(reg_dst, 4));
    str(reg_acc2, ptr(reg_dst, 8));
    str(reg_acc3, ptr(reg_dst, 12));

    postamble();
}

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
    const WReg reg_acc6 = w10;
    const WReg reg_acc7 = w11;

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

jit_int8_brgemm_kernel_1x8_udot::jit_int8_brgemm_kernel_1x8_udot() : jit_generator() {}

void jit_int8_brgemm_kernel_1x8_udot::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_1x8_udot::generate() {
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
    const WReg reg_mask = w27;

    const WReg reg_acc0 = w4;
    const WReg reg_acc1 = w5;
    const WReg reg_acc2 = w6;
    const WReg reg_acc3 = w7;
    const WReg reg_acc4 = w8;
    const WReg reg_acc5 = w9;
    const WReg reg_acc6 = w10;
    const WReg reg_acc7 = w11;

    const VReg16B v_src = VReg16B(0);
    const VReg16B v_w0 = VReg16B(1);
    const VReg16B v_w1 = VReg16B(2);
    const VReg16B v_w2 = VReg16B(3);
    const VReg16B v_w3 = VReg16B(4);
    const VReg16B v_w4 = VReg16B(5);
    const VReg16B v_w5 = VReg16B(6);
    const VReg16B v_w6 = VReg16B(7);
    const VReg16B v_w7 = VReg16B(8);
    const VReg16B v_mask = VReg16B(9);

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

    mov(reg_mask, 0x80);
    dup(v_mask, reg_mask);

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

    eor(v_src, v_src, v_mask);

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

    eor(v_src, v_src, v_mask);

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

    eor(v_src, v_src, v_mask);

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
    ldrb(reg_a, ptr(reg_src));
    sub(reg_a, reg_a, 128);
    ldrsb(reg_b0, ptr(reg_w0));
    ldrsb(reg_b1, ptr(reg_w1));
    ldrsb(reg_b2, ptr(reg_w2));
    ldrsb(reg_b3, ptr(reg_w3));
    ldrsb(reg_b4, ptr(reg_w4));
    ldrsb(reg_b5, ptr(reg_w5));
    ldrsb(reg_b6, ptr(reg_w6));
    ldrsb(reg_b7, ptr(reg_w7));

    add(reg_src, reg_src, 1);
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

jit_int8_brgemm_kernel_1x8_dot_packed::jit_int8_brgemm_kernel_1x8_dot_packed() : jit_generator() {}

void jit_int8_brgemm_kernel_1x8_dot_packed::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_1x8_dot_packed::generate() {
    preamble();

    const XReg reg_src = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_accum = abi_param6;

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
    const WReg reg_acc6 = w10;
    const WReg reg_acc7 = w11;

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

    Label loop64;
    Label check32;
    Label loop32;
    Label loop16;
    Label tail;
    Label tail_loop;
    Label done;
    Label store;

    cmp(reg_k, 64);
    b(LT, check32);

    L(loop64);
    ld1(v_src, ptr(reg_src));
    ld1(VReg16BList(v_w0, v_w3), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    ld1(VReg16BList(v_w4, v_w7), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    add(reg_src, reg_src, 16);
    sdot(v_acc0, v_src, v_w0);
    sdot(v_acc1, v_src, v_w1);
    sdot(v_acc2, v_src, v_w2);
    sdot(v_acc3, v_src, v_w3);
    sdot(v_acc4, v_src, v_w4);
    sdot(v_acc5, v_src, v_w5);
    sdot(v_acc6, v_src, v_w6);
    sdot(v_acc7, v_src, v_w7);

    ld1(v_src, ptr(reg_src));
    ld1(VReg16BList(v_w0, v_w3), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    ld1(VReg16BList(v_w4, v_w7), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    add(reg_src, reg_src, 16);
    sdot(v_acc0, v_src, v_w0);
    sdot(v_acc1, v_src, v_w1);
    sdot(v_acc2, v_src, v_w2);
    sdot(v_acc3, v_src, v_w3);
    sdot(v_acc4, v_src, v_w4);
    sdot(v_acc5, v_src, v_w5);
    sdot(v_acc6, v_src, v_w6);
    sdot(v_acc7, v_src, v_w7);

    ld1(v_src, ptr(reg_src));
    ld1(VReg16BList(v_w0, v_w3), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    ld1(VReg16BList(v_w4, v_w7), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    add(reg_src, reg_src, 16);
    sdot(v_acc0, v_src, v_w0);
    sdot(v_acc1, v_src, v_w1);
    sdot(v_acc2, v_src, v_w2);
    sdot(v_acc3, v_src, v_w3);
    sdot(v_acc4, v_src, v_w4);
    sdot(v_acc5, v_src, v_w5);
    sdot(v_acc6, v_src, v_w6);
    sdot(v_acc7, v_src, v_w7);

    ld1(v_src, ptr(reg_src));
    ld1(VReg16BList(v_w0, v_w3), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    ld1(VReg16BList(v_w4, v_w7), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    add(reg_src, reg_src, 16);
    sdot(v_acc0, v_src, v_w0);
    sdot(v_acc1, v_src, v_w1);
    sdot(v_acc2, v_src, v_w2);
    sdot(v_acc3, v_src, v_w3);
    sdot(v_acc4, v_src, v_w4);
    sdot(v_acc5, v_src, v_w5);
    sdot(v_acc6, v_src, v_w6);
    sdot(v_acc7, v_src, v_w7);

    sub(reg_k, reg_k, 64);
    cmp(reg_k, 64);
    b(GE, loop64);

    L(check32);
    cmp(reg_k, 32);
    b(LT, loop16);

    L(loop32);
    ld1(v_src, ptr(reg_src));
    ld1(VReg16BList(v_w0, v_w3), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    ld1(VReg16BList(v_w4, v_w7), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    add(reg_src, reg_src, 16);
    sdot(v_acc0, v_src, v_w0);
    sdot(v_acc1, v_src, v_w1);
    sdot(v_acc2, v_src, v_w2);
    sdot(v_acc3, v_src, v_w3);
    sdot(v_acc4, v_src, v_w4);
    sdot(v_acc5, v_src, v_w5);
    sdot(v_acc6, v_src, v_w6);
    sdot(v_acc7, v_src, v_w7);

    ld1(v_src, ptr(reg_src));
    ld1(VReg16BList(v_w0, v_w3), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    ld1(VReg16BList(v_w4, v_w7), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    add(reg_src, reg_src, 16);
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
    ld1(VReg16BList(v_w0, v_w3), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    ld1(VReg16BList(v_w4, v_w7), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    add(reg_src, reg_src, 16);
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
    ldrsb(reg_b0, ptr(reg_wei));
    ldrsb(reg_b1, ptr(reg_wei, 1));
    ldrsb(reg_b2, ptr(reg_wei, 2));
    ldrsb(reg_b3, ptr(reg_wei, 3));
    ldrsb(reg_b4, ptr(reg_wei, 4));
    ldrsb(reg_b5, ptr(reg_wei, 5));
    ldrsb(reg_b6, ptr(reg_wei, 6));
    ldrsb(reg_b7, ptr(reg_wei, 7));
    add(reg_src, reg_src, 1);
    add(reg_wei, reg_wei, 8);

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

jit_int8_brgemm_kernel_1x8_udot_packed::jit_int8_brgemm_kernel_1x8_udot_packed() : jit_generator() {}

void jit_int8_brgemm_kernel_1x8_udot_packed::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_1x8_udot_packed::generate() {
    preamble();

    const XReg reg_src = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_accum = abi_param6;

    const WReg reg_a = w26;
    const WReg reg_b0 = w18;
    const WReg reg_b1 = w19;
    const WReg reg_b2 = w20;
    const WReg reg_b3 = w21;
    const WReg reg_b4 = w22;
    const WReg reg_b5 = w23;
    const WReg reg_b6 = w24;
    const WReg reg_b7 = w25;
    const WReg reg_mask = w27;

    const WReg reg_acc0 = w4;
    const WReg reg_acc1 = w5;
    const WReg reg_acc2 = w6;
    const WReg reg_acc3 = w7;
    const WReg reg_acc4 = w8;
    const WReg reg_acc5 = w9;
    const WReg reg_acc6 = w10;
    const WReg reg_acc7 = w11;

    const VReg16B v_src = VReg16B(0);
    const VReg16B v_w0 = VReg16B(1);
    const VReg16B v_w1 = VReg16B(2);
    const VReg16B v_w2 = VReg16B(3);
    const VReg16B v_w3 = VReg16B(4);
    const VReg16B v_w4 = VReg16B(5);
    const VReg16B v_w5 = VReg16B(6);
    const VReg16B v_w6 = VReg16B(7);
    const VReg16B v_w7 = VReg16B(8);
    const VReg16B v_mask = VReg16B(9);

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

    mov(reg_mask, 0x80);
    dup(v_mask, reg_mask);

    eor(v_acc0b, v_acc0b, v_acc0b);
    eor(v_acc1b, v_acc1b, v_acc1b);
    eor(v_acc2b, v_acc2b, v_acc2b);
    eor(v_acc3b, v_acc3b, v_acc3b);
    eor(v_acc4b, v_acc4b, v_acc4b);
    eor(v_acc5b, v_acc5b, v_acc5b);
    eor(v_acc6b, v_acc6b, v_acc6b);
    eor(v_acc7b, v_acc7b, v_acc7b);

    Label loop64;
    Label check32;
    Label loop32;
    Label loop16;
    Label tail;
    Label tail_loop;
    Label done;
    Label store;

    cmp(reg_k, 64);
    b(LT, check32);

    L(loop64);
    ld1(v_src, ptr(reg_src));
    ld1(VReg16BList(v_w0, v_w3), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    ld1(VReg16BList(v_w4, v_w7), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    eor(v_src, v_src, v_mask);
    add(reg_src, reg_src, 16);
    sdot(v_acc0, v_src, v_w0);
    sdot(v_acc1, v_src, v_w1);
    sdot(v_acc2, v_src, v_w2);
    sdot(v_acc3, v_src, v_w3);
    sdot(v_acc4, v_src, v_w4);
    sdot(v_acc5, v_src, v_w5);
    sdot(v_acc6, v_src, v_w6);
    sdot(v_acc7, v_src, v_w7);

    ld1(v_src, ptr(reg_src));
    ld1(VReg16BList(v_w0, v_w3), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    ld1(VReg16BList(v_w4, v_w7), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    eor(v_src, v_src, v_mask);
    add(reg_src, reg_src, 16);
    sdot(v_acc0, v_src, v_w0);
    sdot(v_acc1, v_src, v_w1);
    sdot(v_acc2, v_src, v_w2);
    sdot(v_acc3, v_src, v_w3);
    sdot(v_acc4, v_src, v_w4);
    sdot(v_acc5, v_src, v_w5);
    sdot(v_acc6, v_src, v_w6);
    sdot(v_acc7, v_src, v_w7);

    ld1(v_src, ptr(reg_src));
    ld1(VReg16BList(v_w0, v_w3), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    ld1(VReg16BList(v_w4, v_w7), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    eor(v_src, v_src, v_mask);
    add(reg_src, reg_src, 16);
    sdot(v_acc0, v_src, v_w0);
    sdot(v_acc1, v_src, v_w1);
    sdot(v_acc2, v_src, v_w2);
    sdot(v_acc3, v_src, v_w3);
    sdot(v_acc4, v_src, v_w4);
    sdot(v_acc5, v_src, v_w5);
    sdot(v_acc6, v_src, v_w6);
    sdot(v_acc7, v_src, v_w7);

    ld1(v_src, ptr(reg_src));
    ld1(VReg16BList(v_w0, v_w3), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    ld1(VReg16BList(v_w4, v_w7), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    eor(v_src, v_src, v_mask);
    add(reg_src, reg_src, 16);
    sdot(v_acc0, v_src, v_w0);
    sdot(v_acc1, v_src, v_w1);
    sdot(v_acc2, v_src, v_w2);
    sdot(v_acc3, v_src, v_w3);
    sdot(v_acc4, v_src, v_w4);
    sdot(v_acc5, v_src, v_w5);
    sdot(v_acc6, v_src, v_w6);
    sdot(v_acc7, v_src, v_w7);

    sub(reg_k, reg_k, 64);
    cmp(reg_k, 64);
    b(GE, loop64);

    L(check32);
    cmp(reg_k, 32);
    b(LT, loop16);

    L(loop32);
    ld1(v_src, ptr(reg_src));
    ld1(VReg16BList(v_w0, v_w3), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    ld1(VReg16BList(v_w4, v_w7), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    eor(v_src, v_src, v_mask);
    add(reg_src, reg_src, 16);
    sdot(v_acc0, v_src, v_w0);
    sdot(v_acc1, v_src, v_w1);
    sdot(v_acc2, v_src, v_w2);
    sdot(v_acc3, v_src, v_w3);
    sdot(v_acc4, v_src, v_w4);
    sdot(v_acc5, v_src, v_w5);
    sdot(v_acc6, v_src, v_w6);
    sdot(v_acc7, v_src, v_w7);

    ld1(v_src, ptr(reg_src));
    ld1(VReg16BList(v_w0, v_w3), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    ld1(VReg16BList(v_w4, v_w7), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    eor(v_src, v_src, v_mask);
    add(reg_src, reg_src, 16);
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
    ld1(VReg16BList(v_w0, v_w3), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    ld1(VReg16BList(v_w4, v_w7), ptr(reg_wei));
    add(reg_wei, reg_wei, 64);
    eor(v_src, v_src, v_mask);
    add(reg_src, reg_src, 16);
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
    ldrb(reg_a, ptr(reg_src));
    sub(reg_a, reg_a, 128);
    ldrsb(reg_b0, ptr(reg_wei));
    ldrsb(reg_b1, ptr(reg_wei, 1));
    ldrsb(reg_b2, ptr(reg_wei, 2));
    ldrsb(reg_b3, ptr(reg_wei, 3));
    ldrsb(reg_b4, ptr(reg_wei, 4));
    ldrsb(reg_b5, ptr(reg_wei, 5));
    ldrsb(reg_b6, ptr(reg_wei, 6));
    ldrsb(reg_b7, ptr(reg_wei, 7));
    add(reg_src, reg_src, 1);
    add(reg_wei, reg_wei, 8);

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

jit_int8_brgemm_kernel_4x4_dot::jit_int8_brgemm_kernel_4x4_dot() : jit_generator() {}

void jit_int8_brgemm_kernel_4x4_dot::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_4x4_dot::generate() {
    preamble();

    if (false && (mayiuse(sve_512) || mayiuse(sve_256) || mayiuse(sve_128))) {
        const XReg reg_srcs = abi_param1;
        const XReg reg_wei = abi_param2;
        const XReg reg_dst = abi_param3;
        const XReg reg_k = abi_param4;
        const XReg reg_ldb = abi_param5;
        const XReg reg_ldc = abi_param6;
        const XReg reg_accum = abi_param7;

        const XReg reg_src0 = x10;
        const XReg reg_src1 = x11;
        const XReg reg_src2 = x12;
        const XReg reg_src3 = x13;

        const XReg reg_w0 = x14;
        const XReg reg_w1 = x15;
        const XReg reg_w2 = x16;
        const XReg reg_w3 = x17;

        const XReg reg_c0 = x18;
        const XReg reg_c1 = x19;
        const XReg reg_c2 = x20;
        const XReg reg_c3 = x21;

        const XReg reg_vlen = x22;
        const XReg reg_k_iter = x23;
        const XReg reg_zero = x24;

        const PReg p_all = PReg(0);
        const PReg p_tail = PReg(1);

        const ZRegB z_src0 = ZRegB(0);
        const ZRegB z_src1 = ZRegB(1);
        const ZRegB z_src2 = ZRegB(2);
        const ZRegB z_src3 = ZRegB(3);
        const ZRegB z_w0 = ZRegB(4);
        const ZRegB z_w1 = ZRegB(5);
        const ZRegB z_w2 = ZRegB(6);
        const ZRegB z_w3 = ZRegB(7);

        const ZRegS z_acc00 = ZRegS(16);
        const ZRegS z_acc01 = ZRegS(17);
        const ZRegS z_acc02 = ZRegS(18);
        const ZRegS z_acc03 = ZRegS(19);
        const ZRegS z_acc10 = ZRegS(20);
        const ZRegS z_acc11 = ZRegS(21);
        const ZRegS z_acc12 = ZRegS(22);
        const ZRegS z_acc13 = ZRegS(23);
        const ZRegS z_acc20 = ZRegS(24);
        const ZRegS z_acc21 = ZRegS(25);
        const ZRegS z_acc22 = ZRegS(26);
        const ZRegS z_acc23 = ZRegS(27);
        const ZRegS z_acc30 = ZRegS(28);
        const ZRegS z_acc31 = ZRegS(29);
        const ZRegS z_acc32 = ZRegS(30);
        const ZRegS z_acc33 = ZRegS(31);

        ptrue(p_all.b);
        cntb(reg_vlen);
        mov(reg_zero, 0);

        dup(z_acc00, 0);
        dup(z_acc01, 0);
        dup(z_acc02, 0);
        dup(z_acc03, 0);
        dup(z_acc10, 0);
        dup(z_acc11, 0);
        dup(z_acc12, 0);
        dup(z_acc13, 0);
        dup(z_acc20, 0);
        dup(z_acc21, 0);
        dup(z_acc22, 0);
        dup(z_acc23, 0);
        dup(z_acc30, 0);
        dup(z_acc31, 0);
        dup(z_acc32, 0);
        dup(z_acc33, 0);

        ldr(reg_src0, ptr(reg_srcs));
        ldr(reg_src1, ptr(reg_srcs, 8));
        ldr(reg_src2, ptr(reg_srcs, 16));
        ldr(reg_src3, ptr(reg_srcs, 24));

        mov(reg_w0, reg_wei);
        add(reg_w1, reg_w0, reg_ldb);
        add(reg_w2, reg_w1, reg_ldb);
        add(reg_w3, reg_w2, reg_ldb);

        mov(reg_c0, reg_dst);
        add(reg_c1, reg_c0, reg_ldc);
        add(reg_c2, reg_c1, reg_ldc);
        add(reg_c3, reg_c2, reg_ldc);

        mov(reg_k_iter, reg_k);

        Label loop;
        Label tail;
        Label reduce_store;

        cbz(reg_k_iter, reduce_store);
        cmp(reg_k_iter, reg_vlen);
        b(LT, tail);

        L(loop);
        ld1b(z_src0, p_all, ptr(reg_src0));
        ld1b(z_src1, p_all, ptr(reg_src1));
        ld1b(z_src2, p_all, ptr(reg_src2));
        ld1b(z_src3, p_all, ptr(reg_src3));
        ld1b(z_w0, p_all, ptr(reg_w0));
        ld1b(z_w1, p_all, ptr(reg_w1));
        ld1b(z_w2, p_all, ptr(reg_w2));
        ld1b(z_w3, p_all, ptr(reg_w3));

        sdot(z_acc00, z_src0, z_w0);
        sdot(z_acc01, z_src0, z_w1);
        sdot(z_acc02, z_src0, z_w2);
        sdot(z_acc03, z_src0, z_w3);

        sdot(z_acc10, z_src1, z_w0);
        sdot(z_acc11, z_src1, z_w1);
        sdot(z_acc12, z_src1, z_w2);
        sdot(z_acc13, z_src1, z_w3);

        sdot(z_acc20, z_src2, z_w0);
        sdot(z_acc21, z_src2, z_w1);
        sdot(z_acc22, z_src2, z_w2);
        sdot(z_acc23, z_src2, z_w3);

        sdot(z_acc30, z_src3, z_w0);
        sdot(z_acc31, z_src3, z_w1);
        sdot(z_acc32, z_src3, z_w2);
        sdot(z_acc33, z_src3, z_w3);

        add(reg_src0, reg_src0, reg_vlen);
        add(reg_src1, reg_src1, reg_vlen);
        add(reg_src2, reg_src2, reg_vlen);
        add(reg_src3, reg_src3, reg_vlen);
        add(reg_w0, reg_w0, reg_vlen);
        add(reg_w1, reg_w1, reg_vlen);
        add(reg_w2, reg_w2, reg_vlen);
        add(reg_w3, reg_w3, reg_vlen);

        sub(reg_k_iter, reg_k_iter, reg_vlen);
        cmp(reg_k_iter, reg_vlen);
        b(GE, loop);

        L(tail);
        cbz(reg_k_iter, reduce_store);
        whilelt(p_tail.b, reg_zero, reg_k_iter);
        ld1b(z_src0, p_tail, ptr(reg_src0));
        ld1b(z_src1, p_tail, ptr(reg_src1));
        ld1b(z_src2, p_tail, ptr(reg_src2));
        ld1b(z_src3, p_tail, ptr(reg_src3));
        ld1b(z_w0, p_tail, ptr(reg_w0));
        ld1b(z_w1, p_tail, ptr(reg_w1));
        ld1b(z_w2, p_tail, ptr(reg_w2));
        ld1b(z_w3, p_tail, ptr(reg_w3));

        sdot(z_acc00, z_src0, z_w0);
        sdot(z_acc01, z_src0, z_w1);
        sdot(z_acc02, z_src0, z_w2);
        sdot(z_acc03, z_src0, z_w3);

        sdot(z_acc10, z_src1, z_w0);
        sdot(z_acc11, z_src1, z_w1);
        sdot(z_acc12, z_src1, z_w2);
        sdot(z_acc13, z_src1, z_w3);

        sdot(z_acc20, z_src2, z_w0);
        sdot(z_acc21, z_src2, z_w1);
        sdot(z_acc22, z_src2, z_w2);
        sdot(z_acc23, z_src2, z_w3);

        sdot(z_acc30, z_src3, z_w0);
        sdot(z_acc31, z_src3, z_w1);
        sdot(z_acc32, z_src3, z_w2);
        sdot(z_acc33, z_src3, z_w3);

        L(reduce_store);
        const DReg d_tmp = DReg(0);
        const VReg1D v_tmp_d(d_tmp.getIdx());
        const XReg reg_tmp64 = x9;
        const WReg reg_tmp = w9;
        const WReg reg_c_acc = w0;
        const WReg reg_accum_w(reg_accum.getIdx());

        saddv(d_tmp, p_all, z_acc00);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c0));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c0));
        saddv(d_tmp, p_all, z_acc01);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c0, 4));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c0, 4));
        saddv(d_tmp, p_all, z_acc02);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c0, 8));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c0, 8));
        saddv(d_tmp, p_all, z_acc03);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c0, 12));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c0, 12));

        saddv(d_tmp, p_all, z_acc10);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c1));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c1));
        saddv(d_tmp, p_all, z_acc11);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c1, 4));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c1, 4));
        saddv(d_tmp, p_all, z_acc12);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c1, 8));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c1, 8));
        saddv(d_tmp, p_all, z_acc13);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c1, 12));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c1, 12));

        saddv(d_tmp, p_all, z_acc20);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c2));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c2));
        saddv(d_tmp, p_all, z_acc21);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c2, 4));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c2, 4));
        saddv(d_tmp, p_all, z_acc22);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c2, 8));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c2, 8));
        saddv(d_tmp, p_all, z_acc23);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c2, 12));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c2, 12));

        saddv(d_tmp, p_all, z_acc30);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c3));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c3));
        saddv(d_tmp, p_all, z_acc31);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c3, 4));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c3, 4));
        saddv(d_tmp, p_all, z_acc32);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c3, 8));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c3, 8));
        saddv(d_tmp, p_all, z_acc33);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c3, 12));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c3, 12));

        postamble();
        return;
    }

    const XReg reg_srcs = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_ldb = abi_param5;
    const XReg reg_ldc = abi_param6;
    const XReg reg_accum = abi_param7;

    const XReg reg_src0 = x10;
    const XReg reg_src1 = x11;
    const XReg reg_src2 = x12;
    const XReg reg_src3 = x13;

    const XReg reg_w0 = x14;
    const XReg reg_w1 = x15;
    const XReg reg_w2 = x16;
    const XReg reg_w3 = x17;

    const XReg reg_c0 = x18;
    const XReg reg_c1 = x19;
    const XReg reg_c2 = x20;
    const XReg reg_c3 = x21;

    const VReg16B v_src0 = VReg16B(0);
    const VReg16B v_src1 = VReg16B(1);
    const VReg16B v_src2 = VReg16B(2);
    const VReg16B v_src3 = VReg16B(3);
    const VReg16B v_w0 = VReg16B(4);
    const VReg16B v_w1 = VReg16B(5);
    const VReg16B v_w2 = VReg16B(6);
    const VReg16B v_w3 = VReg16B(7);

    const VReg4S v_acc00 = VReg4S(16);
    const VReg4S v_acc01 = VReg4S(17);
    const VReg4S v_acc02 = VReg4S(18);
    const VReg4S v_acc03 = VReg4S(19);
    const VReg4S v_acc10 = VReg4S(20);
    const VReg4S v_acc11 = VReg4S(21);
    const VReg4S v_acc12 = VReg4S(22);
    const VReg4S v_acc13 = VReg4S(23);
    const VReg4S v_acc20 = VReg4S(24);
    const VReg4S v_acc21 = VReg4S(25);
    const VReg4S v_acc22 = VReg4S(26);
    const VReg4S v_acc23 = VReg4S(27);
    const VReg4S v_acc30 = VReg4S(28);
    const VReg4S v_acc31 = VReg4S(29);
    const VReg4S v_acc32 = VReg4S(30);
    const VReg4S v_acc33 = VReg4S(31);

    const VReg16B v_acc00b(v_acc00.getIdx());
    const VReg16B v_acc01b(v_acc01.getIdx());
    const VReg16B v_acc02b(v_acc02.getIdx());
    const VReg16B v_acc03b(v_acc03.getIdx());
    const VReg16B v_acc10b(v_acc10.getIdx());
    const VReg16B v_acc11b(v_acc11.getIdx());
    const VReg16B v_acc12b(v_acc12.getIdx());
    const VReg16B v_acc13b(v_acc13.getIdx());
    const VReg16B v_acc20b(v_acc20.getIdx());
    const VReg16B v_acc21b(v_acc21.getIdx());
    const VReg16B v_acc22b(v_acc22.getIdx());
    const VReg16B v_acc23b(v_acc23.getIdx());
    const VReg16B v_acc30b(v_acc30.getIdx());
    const VReg16B v_acc31b(v_acc31.getIdx());
    const VReg16B v_acc32b(v_acc32.getIdx());
    const VReg16B v_acc33b(v_acc33.getIdx());

    eor(v_acc00b, v_acc00b, v_acc00b);
    eor(v_acc01b, v_acc01b, v_acc01b);
    eor(v_acc02b, v_acc02b, v_acc02b);
    eor(v_acc03b, v_acc03b, v_acc03b);
    eor(v_acc10b, v_acc10b, v_acc10b);
    eor(v_acc11b, v_acc11b, v_acc11b);
    eor(v_acc12b, v_acc12b, v_acc12b);
    eor(v_acc13b, v_acc13b, v_acc13b);
    eor(v_acc20b, v_acc20b, v_acc20b);
    eor(v_acc21b, v_acc21b, v_acc21b);
    eor(v_acc22b, v_acc22b, v_acc22b);
    eor(v_acc23b, v_acc23b, v_acc23b);
    eor(v_acc30b, v_acc30b, v_acc30b);
    eor(v_acc31b, v_acc31b, v_acc31b);
    eor(v_acc32b, v_acc32b, v_acc32b);
    eor(v_acc33b, v_acc33b, v_acc33b);

    ldr(reg_src0, ptr(reg_srcs));
    ldr(reg_src1, ptr(reg_srcs, 8));
    ldr(reg_src2, ptr(reg_srcs, 16));
    ldr(reg_src3, ptr(reg_srcs, 24));

    mov(reg_w0, reg_wei);
    add(reg_w1, reg_w0, reg_ldb);
    add(reg_w2, reg_w1, reg_ldb);
    add(reg_w3, reg_w2, reg_ldb);

    mov(reg_c0, reg_dst);
    add(reg_c1, reg_c0, reg_ldc);
    add(reg_c2, reg_c1, reg_ldc);
    add(reg_c3, reg_c2, reg_ldc);

    Label loop32;
    Label loop16;
    Label reduce_store;
    Label tail_loop;
    Label done;

    cmp(reg_k, 32);
    b(LT, loop16);

    L(loop32);
    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(v_src2, ptr(reg_src2));
    ld1(v_src3, ptr(reg_src3));
    ld1(v_w0, ptr(reg_w0));
    ld1(v_w1, ptr(reg_w1));
    ld1(v_w2, ptr(reg_w2));
    ld1(v_w3, ptr(reg_w3));

    sdot(v_acc00, v_src0, v_w0);
    sdot(v_acc01, v_src0, v_w1);
    sdot(v_acc02, v_src0, v_w2);
    sdot(v_acc03, v_src0, v_w3);

    sdot(v_acc10, v_src1, v_w0);
    sdot(v_acc11, v_src1, v_w1);
    sdot(v_acc12, v_src1, v_w2);
    sdot(v_acc13, v_src1, v_w3);

    sdot(v_acc20, v_src2, v_w0);
    sdot(v_acc21, v_src2, v_w1);
    sdot(v_acc22, v_src2, v_w2);
    sdot(v_acc23, v_src2, v_w3);

    sdot(v_acc30, v_src3, v_w0);
    sdot(v_acc31, v_src3, v_w1);
    sdot(v_acc32, v_src3, v_w2);
    sdot(v_acc33, v_src3, v_w3);

    ldr(QReg(v_src0.getIdx()), ptr(reg_src0, 16));
    ldr(QReg(v_src1.getIdx()), ptr(reg_src1, 16));
    ldr(QReg(v_src2.getIdx()), ptr(reg_src2, 16));
    ldr(QReg(v_src3.getIdx()), ptr(reg_src3, 16));
    ldr(QReg(v_w0.getIdx()), ptr(reg_w0, 16));
    ldr(QReg(v_w1.getIdx()), ptr(reg_w1, 16));
    ldr(QReg(v_w2.getIdx()), ptr(reg_w2, 16));
    ldr(QReg(v_w3.getIdx()), ptr(reg_w3, 16));

    sdot(v_acc00, v_src0, v_w0);
    sdot(v_acc01, v_src0, v_w1);
    sdot(v_acc02, v_src0, v_w2);
    sdot(v_acc03, v_src0, v_w3);

    sdot(v_acc10, v_src1, v_w0);
    sdot(v_acc11, v_src1, v_w1);
    sdot(v_acc12, v_src1, v_w2);
    sdot(v_acc13, v_src1, v_w3);

    sdot(v_acc20, v_src2, v_w0);
    sdot(v_acc21, v_src2, v_w1);
    sdot(v_acc22, v_src2, v_w2);
    sdot(v_acc23, v_src2, v_w3);

    sdot(v_acc30, v_src3, v_w0);
    sdot(v_acc31, v_src3, v_w1);
    sdot(v_acc32, v_src3, v_w2);
    sdot(v_acc33, v_src3, v_w3);

    add(reg_src0, reg_src0, 32);
    add(reg_src1, reg_src1, 32);
    add(reg_src2, reg_src2, 32);
    add(reg_src3, reg_src3, 32);
    add(reg_w0, reg_w0, 32);
    add(reg_w1, reg_w1, 32);
    add(reg_w2, reg_w2, 32);
    add(reg_w3, reg_w3, 32);

    sub(reg_k, reg_k, 32);
    cmp(reg_k, 32);
    b(GE, loop32);

    L(loop16);
    cmp(reg_k, 16);
    b(LT, reduce_store);
    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(v_src2, ptr(reg_src2));
    ld1(v_src3, ptr(reg_src3));
    ld1(v_w0, ptr(reg_w0));
    ld1(v_w1, ptr(reg_w1));
    ld1(v_w2, ptr(reg_w2));
    ld1(v_w3, ptr(reg_w3));

    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
    add(reg_src2, reg_src2, 16);
    add(reg_src3, reg_src3, 16);
    add(reg_w0, reg_w0, 16);
    add(reg_w1, reg_w1, 16);
    add(reg_w2, reg_w2, 16);
    add(reg_w3, reg_w3, 16);

    sdot(v_acc00, v_src0, v_w0);
    sdot(v_acc01, v_src0, v_w1);
    sdot(v_acc02, v_src0, v_w2);
    sdot(v_acc03, v_src0, v_w3);

    sdot(v_acc10, v_src1, v_w0);
    sdot(v_acc11, v_src1, v_w1);
    sdot(v_acc12, v_src1, v_w2);
    sdot(v_acc13, v_src1, v_w3);

    sdot(v_acc20, v_src2, v_w0);
    sdot(v_acc21, v_src2, v_w1);
    sdot(v_acc22, v_src2, v_w2);
    sdot(v_acc23, v_src2, v_w3);

    sdot(v_acc30, v_src3, v_w0);
    sdot(v_acc31, v_src3, v_w1);
    sdot(v_acc32, v_src3, v_w2);
    sdot(v_acc33, v_src3, v_w3);

    sub(reg_k, reg_k, 16);

    L(reduce_store);
    const VReg v_tmp(0);
    const SReg s_tmp(v_tmp.getIdx());
    const WReg reg_tmp = w9;
    const WReg reg_c_acc = w0;
    const WReg reg_accum_w(reg_accum.getIdx());

    addv(s_tmp, v_acc00);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0));
    addv(s_tmp, v_acc01);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 4));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 4));
    addv(s_tmp, v_acc02);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 8));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 8));
    addv(s_tmp, v_acc03);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 12));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 12));

    addv(s_tmp, v_acc10);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1));
    addv(s_tmp, v_acc11);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 4));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 4));
    addv(s_tmp, v_acc12);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 8));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 8));
    addv(s_tmp, v_acc13);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 12));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 12));

    addv(s_tmp, v_acc20);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c2));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c2));
    addv(s_tmp, v_acc21);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c2, 4));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 4));
    addv(s_tmp, v_acc22);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c2, 8));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 8));
    addv(s_tmp, v_acc23);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c2, 12));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 12));

    addv(s_tmp, v_acc30);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c3));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c3));
    addv(s_tmp, v_acc31);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c3, 4));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 4));
    addv(s_tmp, v_acc32);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c3, 8));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 8));
    addv(s_tmp, v_acc33);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c3, 12));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 12));

    cbz(reg_k, done);

    const WReg reg_a0 = w22;
    const WReg reg_a1 = w23;
    const WReg reg_a2 = w24;
    const WReg reg_a3 = w25;
    const WReg reg_b0 = w26;
    const WReg reg_b1 = w27;
    const WReg reg_b2 = w28;
    const WReg reg_b3 = w8;

    L(tail_loop);
    ldrsb(reg_a0, ptr(reg_src0));
    ldrsb(reg_a1, ptr(reg_src1));
    ldrsb(reg_a2, ptr(reg_src2));
    ldrsb(reg_a3, ptr(reg_src3));
    ldrsb(reg_b0, ptr(reg_w0));
    ldrsb(reg_b1, ptr(reg_w1));
    ldrsb(reg_b2, ptr(reg_w2));
    ldrsb(reg_b3, ptr(reg_w3));

    add(reg_src0, reg_src0, 1);
    add(reg_src1, reg_src1, 1);
    add(reg_src2, reg_src2, 1);
    add(reg_src3, reg_src3, 1);
    add(reg_w0, reg_w0, 1);
    add(reg_w1, reg_w1, 1);
    add(reg_w2, reg_w2, 1);
    add(reg_w3, reg_w3, 1);

    ldr(reg_tmp, ptr(reg_c0));
    madd(reg_tmp, reg_a0, reg_b0, reg_tmp);
    str(reg_tmp, ptr(reg_c0));
    ldr(reg_tmp, ptr(reg_c0, 4));
    madd(reg_tmp, reg_a0, reg_b1, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 4));
    ldr(reg_tmp, ptr(reg_c0, 8));
    madd(reg_tmp, reg_a0, reg_b2, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 8));
    ldr(reg_tmp, ptr(reg_c0, 12));
    madd(reg_tmp, reg_a0, reg_b3, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 12));

    ldr(reg_tmp, ptr(reg_c1));
    madd(reg_tmp, reg_a1, reg_b0, reg_tmp);
    str(reg_tmp, ptr(reg_c1));
    ldr(reg_tmp, ptr(reg_c1, 4));
    madd(reg_tmp, reg_a1, reg_b1, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 4));
    ldr(reg_tmp, ptr(reg_c1, 8));
    madd(reg_tmp, reg_a1, reg_b2, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 8));
    ldr(reg_tmp, ptr(reg_c1, 12));
    madd(reg_tmp, reg_a1, reg_b3, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 12));

    ldr(reg_tmp, ptr(reg_c2));
    madd(reg_tmp, reg_a2, reg_b0, reg_tmp);
    str(reg_tmp, ptr(reg_c2));
    ldr(reg_tmp, ptr(reg_c2, 4));
    madd(reg_tmp, reg_a2, reg_b1, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 4));
    ldr(reg_tmp, ptr(reg_c2, 8));
    madd(reg_tmp, reg_a2, reg_b2, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 8));
    ldr(reg_tmp, ptr(reg_c2, 12));
    madd(reg_tmp, reg_a2, reg_b3, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 12));

    ldr(reg_tmp, ptr(reg_c3));
    madd(reg_tmp, reg_a3, reg_b0, reg_tmp);
    str(reg_tmp, ptr(reg_c3));
    ldr(reg_tmp, ptr(reg_c3, 4));
    madd(reg_tmp, reg_a3, reg_b1, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 4));
    ldr(reg_tmp, ptr(reg_c3, 8));
    madd(reg_tmp, reg_a3, reg_b2, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 8));
    ldr(reg_tmp, ptr(reg_c3, 12));
    madd(reg_tmp, reg_a3, reg_b3, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 12));

    subs(reg_k, reg_k, 1);
    b(NE, tail_loop);

    L(done);
    postamble();
}

jit_int8_brgemm_kernel_4x4_udot::jit_int8_brgemm_kernel_4x4_udot() : jit_generator() {}

void jit_int8_brgemm_kernel_4x4_udot::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_4x4_udot::generate() {
    preamble();

    if (false && (mayiuse(sve_512) || mayiuse(sve_256) || mayiuse(sve_128))) {
        const XReg reg_srcs = abi_param1;
        const XReg reg_wei = abi_param2;
        const XReg reg_dst = abi_param3;
        const XReg reg_k = abi_param4;
        const XReg reg_ldb = abi_param5;
        const XReg reg_ldc = abi_param6;
        const XReg reg_accum = abi_param7;

        const XReg reg_src0 = x10;
        const XReg reg_src1 = x11;
        const XReg reg_src2 = x12;
        const XReg reg_src3 = x13;

        const XReg reg_w0 = x14;
        const XReg reg_w1 = x15;
        const XReg reg_w2 = x16;
        const XReg reg_w3 = x17;

        const XReg reg_c0 = x18;
        const XReg reg_c1 = x19;
        const XReg reg_c2 = x20;
        const XReg reg_c3 = x21;

        const XReg reg_vlen = x22;
        const XReg reg_k_iter = x23;
        const XReg reg_zero = x24;
        const WReg reg_mask = w10;

        const PReg p_all = PReg(0);
        const PReg p_tail = PReg(1);

        const ZRegB z_src0 = ZRegB(0);
        const ZRegB z_src1 = ZRegB(1);
        const ZRegB z_src2 = ZRegB(2);
        const ZRegB z_src3 = ZRegB(3);
        const ZRegB z_w0 = ZRegB(4);
        const ZRegB z_w1 = ZRegB(5);
        const ZRegB z_w2 = ZRegB(6);
        const ZRegB z_w3 = ZRegB(7);
        const ZRegB z_mask = ZRegB(8);

        const ZRegS z_acc00 = ZRegS(16);
        const ZRegS z_acc01 = ZRegS(17);
        const ZRegS z_acc02 = ZRegS(18);
        const ZRegS z_acc03 = ZRegS(19);
        const ZRegS z_acc10 = ZRegS(20);
        const ZRegS z_acc11 = ZRegS(21);
        const ZRegS z_acc12 = ZRegS(22);
        const ZRegS z_acc13 = ZRegS(23);
        const ZRegS z_acc20 = ZRegS(24);
        const ZRegS z_acc21 = ZRegS(25);
        const ZRegS z_acc22 = ZRegS(26);
        const ZRegS z_acc23 = ZRegS(27);
        const ZRegS z_acc30 = ZRegS(28);
        const ZRegS z_acc31 = ZRegS(29);
        const ZRegS z_acc32 = ZRegS(30);
        const ZRegS z_acc33 = ZRegS(31);

        ptrue(p_all.b);
        cntb(reg_vlen);
        mov(reg_zero, 0);

        mov(reg_mask, 0x80);
        dup(z_mask, reg_mask);

        dup(z_acc00, 0);
        dup(z_acc01, 0);
        dup(z_acc02, 0);
        dup(z_acc03, 0);
        dup(z_acc10, 0);
        dup(z_acc11, 0);
        dup(z_acc12, 0);
        dup(z_acc13, 0);
        dup(z_acc20, 0);
        dup(z_acc21, 0);
        dup(z_acc22, 0);
        dup(z_acc23, 0);
        dup(z_acc30, 0);
        dup(z_acc31, 0);
        dup(z_acc32, 0);
        dup(z_acc33, 0);

        ldr(reg_src0, ptr(reg_srcs));
        ldr(reg_src1, ptr(reg_srcs, 8));
        ldr(reg_src2, ptr(reg_srcs, 16));
        ldr(reg_src3, ptr(reg_srcs, 24));

        mov(reg_w0, reg_wei);
        add(reg_w1, reg_w0, reg_ldb);
        add(reg_w2, reg_w1, reg_ldb);
        add(reg_w3, reg_w2, reg_ldb);

        mov(reg_c0, reg_dst);
        add(reg_c1, reg_c0, reg_ldc);
        add(reg_c2, reg_c1, reg_ldc);
        add(reg_c3, reg_c2, reg_ldc);

        mov(reg_k_iter, reg_k);

        Label loop;
        Label tail;
        Label reduce_store;

        cbz(reg_k_iter, reduce_store);
        cmp(reg_k_iter, reg_vlen);
        b(LT, tail);

        L(loop);
        ld1b(z_src0, p_all, ptr(reg_src0));
        ld1b(z_src1, p_all, ptr(reg_src1));
        ld1b(z_src2, p_all, ptr(reg_src2));
        ld1b(z_src3, p_all, ptr(reg_src3));
        ld1b(z_w0, p_all, ptr(reg_w0));
        ld1b(z_w1, p_all, ptr(reg_w1));
        ld1b(z_w2, p_all, ptr(reg_w2));
        ld1b(z_w3, p_all, ptr(reg_w3));

        eor(z_src0, p_all, z_mask);
        eor(z_src1, p_all, z_mask);
        eor(z_src2, p_all, z_mask);
        eor(z_src3, p_all, z_mask);

        sdot(z_acc00, z_src0, z_w0);
        sdot(z_acc01, z_src0, z_w1);
        sdot(z_acc02, z_src0, z_w2);
        sdot(z_acc03, z_src0, z_w3);

        sdot(z_acc10, z_src1, z_w0);
        sdot(z_acc11, z_src1, z_w1);
        sdot(z_acc12, z_src1, z_w2);
        sdot(z_acc13, z_src1, z_w3);

        sdot(z_acc20, z_src2, z_w0);
        sdot(z_acc21, z_src2, z_w1);
        sdot(z_acc22, z_src2, z_w2);
        sdot(z_acc23, z_src2, z_w3);

        sdot(z_acc30, z_src3, z_w0);
        sdot(z_acc31, z_src3, z_w1);
        sdot(z_acc32, z_src3, z_w2);
        sdot(z_acc33, z_src3, z_w3);

        add(reg_src0, reg_src0, reg_vlen);
        add(reg_src1, reg_src1, reg_vlen);
        add(reg_src2, reg_src2, reg_vlen);
        add(reg_src3, reg_src3, reg_vlen);
        add(reg_w0, reg_w0, reg_vlen);
        add(reg_w1, reg_w1, reg_vlen);
        add(reg_w2, reg_w2, reg_vlen);
        add(reg_w3, reg_w3, reg_vlen);

        sub(reg_k_iter, reg_k_iter, reg_vlen);
        cmp(reg_k_iter, reg_vlen);
        b(GE, loop);

        L(tail);
        cbz(reg_k_iter, reduce_store);
        whilelt(p_tail.b, reg_zero, reg_k_iter);
        ld1b(z_src0, p_tail, ptr(reg_src0));
        ld1b(z_src1, p_tail, ptr(reg_src1));
        ld1b(z_src2, p_tail, ptr(reg_src2));
        ld1b(z_src3, p_tail, ptr(reg_src3));
        ld1b(z_w0, p_tail, ptr(reg_w0));
        ld1b(z_w1, p_tail, ptr(reg_w1));
        ld1b(z_w2, p_tail, ptr(reg_w2));
        ld1b(z_w3, p_tail, ptr(reg_w3));

        eor(z_src0, p_tail, z_mask);
        eor(z_src1, p_tail, z_mask);
        eor(z_src2, p_tail, z_mask);
        eor(z_src3, p_tail, z_mask);

        sdot(z_acc00, z_src0, z_w0);
        sdot(z_acc01, z_src0, z_w1);
        sdot(z_acc02, z_src0, z_w2);
        sdot(z_acc03, z_src0, z_w3);

        sdot(z_acc10, z_src1, z_w0);
        sdot(z_acc11, z_src1, z_w1);
        sdot(z_acc12, z_src1, z_w2);
        sdot(z_acc13, z_src1, z_w3);

        sdot(z_acc20, z_src2, z_w0);
        sdot(z_acc21, z_src2, z_w1);
        sdot(z_acc22, z_src2, z_w2);
        sdot(z_acc23, z_src2, z_w3);

        sdot(z_acc30, z_src3, z_w0);
        sdot(z_acc31, z_src3, z_w1);
        sdot(z_acc32, z_src3, z_w2);
        sdot(z_acc33, z_src3, z_w3);

        L(reduce_store);
        const DReg d_tmp = DReg(0);
        const VReg1D v_tmp_d(d_tmp.getIdx());
        const XReg reg_tmp64 = x9;
        const WReg reg_tmp = w9;
        const WReg reg_c_acc = w0;
        const WReg reg_accum_w(reg_accum.getIdx());

        saddv(d_tmp, p_all, z_acc00);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c0));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c0));
        saddv(d_tmp, p_all, z_acc01);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c0, 4));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c0, 4));
        saddv(d_tmp, p_all, z_acc02);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c0, 8));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c0, 8));
        saddv(d_tmp, p_all, z_acc03);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c0, 12));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c0, 12));

        saddv(d_tmp, p_all, z_acc10);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c1));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c1));
        saddv(d_tmp, p_all, z_acc11);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c1, 4));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c1, 4));
        saddv(d_tmp, p_all, z_acc12);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c1, 8));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c1, 8));
        saddv(d_tmp, p_all, z_acc13);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c1, 12));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c1, 12));

        saddv(d_tmp, p_all, z_acc20);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c2));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c2));
        saddv(d_tmp, p_all, z_acc21);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c2, 4));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c2, 4));
        saddv(d_tmp, p_all, z_acc22);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c2, 8));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c2, 8));
        saddv(d_tmp, p_all, z_acc23);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c2, 12));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c2, 12));

        saddv(d_tmp, p_all, z_acc30);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c3));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c3));
        saddv(d_tmp, p_all, z_acc31);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c3, 4));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c3, 4));
        saddv(d_tmp, p_all, z_acc32);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c3, 8));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c3, 8));
        saddv(d_tmp, p_all, z_acc33);
        umov(reg_tmp64, v_tmp_d[0]);
        ldr(reg_c_acc, ptr(reg_c3, 12));
        madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
        str(reg_tmp, ptr(reg_c3, 12));

        postamble();
        return;
    }

    const XReg reg_srcs = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_ldb = abi_param5;
    const XReg reg_ldc = abi_param6;
    const XReg reg_accum = abi_param7;

    const XReg reg_src0 = x10;
    const XReg reg_src1 = x11;
    const XReg reg_src2 = x12;
    const XReg reg_src3 = x13;

    const XReg reg_w0 = x14;
    const XReg reg_w1 = x15;
    const XReg reg_w2 = x16;
    const XReg reg_w3 = x17;

    const XReg reg_c0 = x18;
    const XReg reg_c1 = x19;
    const XReg reg_c2 = x20;
    const XReg reg_c3 = x21;

    const VReg16B v_src0 = VReg16B(0);
    const VReg16B v_src1 = VReg16B(1);
    const VReg16B v_src2 = VReg16B(2);
    const VReg16B v_src3 = VReg16B(3);
    const VReg16B v_w0 = VReg16B(4);
    const VReg16B v_w1 = VReg16B(5);
    const VReg16B v_w2 = VReg16B(6);
    const VReg16B v_w3 = VReg16B(7);
    const VReg16B v_mask = VReg16B(8);

    const VReg4S v_acc00 = VReg4S(16);
    const VReg4S v_acc01 = VReg4S(17);
    const VReg4S v_acc02 = VReg4S(18);
    const VReg4S v_acc03 = VReg4S(19);
    const VReg4S v_acc10 = VReg4S(20);
    const VReg4S v_acc11 = VReg4S(21);
    const VReg4S v_acc12 = VReg4S(22);
    const VReg4S v_acc13 = VReg4S(23);
    const VReg4S v_acc20 = VReg4S(24);
    const VReg4S v_acc21 = VReg4S(25);
    const VReg4S v_acc22 = VReg4S(26);
    const VReg4S v_acc23 = VReg4S(27);
    const VReg4S v_acc30 = VReg4S(28);
    const VReg4S v_acc31 = VReg4S(29);
    const VReg4S v_acc32 = VReg4S(30);
    const VReg4S v_acc33 = VReg4S(31);

    const VReg16B v_acc00b(v_acc00.getIdx());
    const VReg16B v_acc01b(v_acc01.getIdx());
    const VReg16B v_acc02b(v_acc02.getIdx());
    const VReg16B v_acc03b(v_acc03.getIdx());
    const VReg16B v_acc10b(v_acc10.getIdx());
    const VReg16B v_acc11b(v_acc11.getIdx());
    const VReg16B v_acc12b(v_acc12.getIdx());
    const VReg16B v_acc13b(v_acc13.getIdx());
    const VReg16B v_acc20b(v_acc20.getIdx());
    const VReg16B v_acc21b(v_acc21.getIdx());
    const VReg16B v_acc22b(v_acc22.getIdx());
    const VReg16B v_acc23b(v_acc23.getIdx());
    const VReg16B v_acc30b(v_acc30.getIdx());
    const VReg16B v_acc31b(v_acc31.getIdx());
    const VReg16B v_acc32b(v_acc32.getIdx());
    const VReg16B v_acc33b(v_acc33.getIdx());

    const WReg reg_mask = w10;
    mov(reg_mask, 0x80);
    dup(v_mask, reg_mask);

    eor(v_acc00b, v_acc00b, v_acc00b);
    eor(v_acc01b, v_acc01b, v_acc01b);
    eor(v_acc02b, v_acc02b, v_acc02b);
    eor(v_acc03b, v_acc03b, v_acc03b);
    eor(v_acc10b, v_acc10b, v_acc10b);
    eor(v_acc11b, v_acc11b, v_acc11b);
    eor(v_acc12b, v_acc12b, v_acc12b);
    eor(v_acc13b, v_acc13b, v_acc13b);
    eor(v_acc20b, v_acc20b, v_acc20b);
    eor(v_acc21b, v_acc21b, v_acc21b);
    eor(v_acc22b, v_acc22b, v_acc22b);
    eor(v_acc23b, v_acc23b, v_acc23b);
    eor(v_acc30b, v_acc30b, v_acc30b);
    eor(v_acc31b, v_acc31b, v_acc31b);
    eor(v_acc32b, v_acc32b, v_acc32b);
    eor(v_acc33b, v_acc33b, v_acc33b);

    ldr(reg_src0, ptr(reg_srcs));
    ldr(reg_src1, ptr(reg_srcs, 8));
    ldr(reg_src2, ptr(reg_srcs, 16));
    ldr(reg_src3, ptr(reg_srcs, 24));

    mov(reg_w0, reg_wei);
    add(reg_w1, reg_w0, reg_ldb);
    add(reg_w2, reg_w1, reg_ldb);
    add(reg_w3, reg_w2, reg_ldb);

    mov(reg_c0, reg_dst);
    add(reg_c1, reg_c0, reg_ldc);
    add(reg_c2, reg_c1, reg_ldc);
    add(reg_c3, reg_c2, reg_ldc);

    Label loop32;
    Label loop16;
    Label reduce_store;
    Label tail_loop;
    Label done;

    cmp(reg_k, 32);
    b(LT, loop16);

    L(loop32);
    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(v_src2, ptr(reg_src2));
    ld1(v_src3, ptr(reg_src3));
    ld1(v_w0, ptr(reg_w0));
    ld1(v_w1, ptr(reg_w1));
    ld1(v_w2, ptr(reg_w2));
    ld1(v_w3, ptr(reg_w3));

    eor(v_src0, v_src0, v_mask);
    eor(v_src1, v_src1, v_mask);
    eor(v_src2, v_src2, v_mask);
    eor(v_src3, v_src3, v_mask);

    sdot(v_acc00, v_src0, v_w0);
    sdot(v_acc01, v_src0, v_w1);
    sdot(v_acc02, v_src0, v_w2);
    sdot(v_acc03, v_src0, v_w3);

    sdot(v_acc10, v_src1, v_w0);
    sdot(v_acc11, v_src1, v_w1);
    sdot(v_acc12, v_src1, v_w2);
    sdot(v_acc13, v_src1, v_w3);

    sdot(v_acc20, v_src2, v_w0);
    sdot(v_acc21, v_src2, v_w1);
    sdot(v_acc22, v_src2, v_w2);
    sdot(v_acc23, v_src2, v_w3);

    sdot(v_acc30, v_src3, v_w0);
    sdot(v_acc31, v_src3, v_w1);
    sdot(v_acc32, v_src3, v_w2);
    sdot(v_acc33, v_src3, v_w3);

    ldr(QReg(v_src0.getIdx()), ptr(reg_src0, 16));
    ldr(QReg(v_src1.getIdx()), ptr(reg_src1, 16));
    ldr(QReg(v_src2.getIdx()), ptr(reg_src2, 16));
    ldr(QReg(v_src3.getIdx()), ptr(reg_src3, 16));
    ldr(QReg(v_w0.getIdx()), ptr(reg_w0, 16));
    ldr(QReg(v_w1.getIdx()), ptr(reg_w1, 16));
    ldr(QReg(v_w2.getIdx()), ptr(reg_w2, 16));
    ldr(QReg(v_w3.getIdx()), ptr(reg_w3, 16));

    eor(v_src0, v_src0, v_mask);
    eor(v_src1, v_src1, v_mask);
    eor(v_src2, v_src2, v_mask);
    eor(v_src3, v_src3, v_mask);

    sdot(v_acc00, v_src0, v_w0);
    sdot(v_acc01, v_src0, v_w1);
    sdot(v_acc02, v_src0, v_w2);
    sdot(v_acc03, v_src0, v_w3);

    sdot(v_acc10, v_src1, v_w0);
    sdot(v_acc11, v_src1, v_w1);
    sdot(v_acc12, v_src1, v_w2);
    sdot(v_acc13, v_src1, v_w3);

    sdot(v_acc20, v_src2, v_w0);
    sdot(v_acc21, v_src2, v_w1);
    sdot(v_acc22, v_src2, v_w2);
    sdot(v_acc23, v_src2, v_w3);

    sdot(v_acc30, v_src3, v_w0);
    sdot(v_acc31, v_src3, v_w1);
    sdot(v_acc32, v_src3, v_w2);
    sdot(v_acc33, v_src3, v_w3);

    add(reg_src0, reg_src0, 32);
    add(reg_src1, reg_src1, 32);
    add(reg_src2, reg_src2, 32);
    add(reg_src3, reg_src3, 32);
    add(reg_w0, reg_w0, 32);
    add(reg_w1, reg_w1, 32);
    add(reg_w2, reg_w2, 32);
    add(reg_w3, reg_w3, 32);

    sub(reg_k, reg_k, 32);
    cmp(reg_k, 32);
    b(GE, loop32);

    L(loop16);
    cmp(reg_k, 16);
    b(LT, reduce_store);
    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(v_src2, ptr(reg_src2));
    ld1(v_src3, ptr(reg_src3));
    ld1(v_w0, ptr(reg_w0));
    ld1(v_w1, ptr(reg_w1));
    ld1(v_w2, ptr(reg_w2));
    ld1(v_w3, ptr(reg_w3));

    eor(v_src0, v_src0, v_mask);
    eor(v_src1, v_src1, v_mask);
    eor(v_src2, v_src2, v_mask);
    eor(v_src3, v_src3, v_mask);

    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
    add(reg_src2, reg_src2, 16);
    add(reg_src3, reg_src3, 16);
    add(reg_w0, reg_w0, 16);
    add(reg_w1, reg_w1, 16);
    add(reg_w2, reg_w2, 16);
    add(reg_w3, reg_w3, 16);

    sdot(v_acc00, v_src0, v_w0);
    sdot(v_acc01, v_src0, v_w1);
    sdot(v_acc02, v_src0, v_w2);
    sdot(v_acc03, v_src0, v_w3);

    sdot(v_acc10, v_src1, v_w0);
    sdot(v_acc11, v_src1, v_w1);
    sdot(v_acc12, v_src1, v_w2);
    sdot(v_acc13, v_src1, v_w3);

    sdot(v_acc20, v_src2, v_w0);
    sdot(v_acc21, v_src2, v_w1);
    sdot(v_acc22, v_src2, v_w2);
    sdot(v_acc23, v_src2, v_w3);

    sdot(v_acc30, v_src3, v_w0);
    sdot(v_acc31, v_src3, v_w1);
    sdot(v_acc32, v_src3, v_w2);
    sdot(v_acc33, v_src3, v_w3);

    sub(reg_k, reg_k, 16);

    L(reduce_store);
    const VReg v_tmp(0);
    const SReg s_tmp(v_tmp.getIdx());
    const WReg reg_tmp = w9;
    const WReg reg_c_acc = w0;
    const WReg reg_accum_w(reg_accum.getIdx());

    addv(s_tmp, v_acc00);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0));
    addv(s_tmp, v_acc01);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 4));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 4));
    addv(s_tmp, v_acc02);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 8));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 8));
    addv(s_tmp, v_acc03);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 12));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 12));

    addv(s_tmp, v_acc10);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1));
    addv(s_tmp, v_acc11);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 4));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 4));
    addv(s_tmp, v_acc12);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 8));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 8));
    addv(s_tmp, v_acc13);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 12));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 12));

    addv(s_tmp, v_acc20);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c2));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c2));
    addv(s_tmp, v_acc21);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c2, 4));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 4));
    addv(s_tmp, v_acc22);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c2, 8));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 8));
    addv(s_tmp, v_acc23);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c2, 12));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 12));

    addv(s_tmp, v_acc30);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c3));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c3));
    addv(s_tmp, v_acc31);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c3, 4));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 4));
    addv(s_tmp, v_acc32);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c3, 8));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 8));
    addv(s_tmp, v_acc33);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c3, 12));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 12));

    cbz(reg_k, done);

    const WReg reg_a0 = w22;
    const WReg reg_a1 = w23;
    const WReg reg_a2 = w24;
    const WReg reg_a3 = w25;
    const WReg reg_b0 = w26;
    const WReg reg_b1 = w27;
    const WReg reg_b2 = w28;
    const WReg reg_b3 = w8;

    L(tail_loop);
    ldrb(reg_a0, ptr(reg_src0));
    ldrb(reg_a1, ptr(reg_src1));
    ldrb(reg_a2, ptr(reg_src2));
    ldrb(reg_a3, ptr(reg_src3));
    sub(reg_a0, reg_a0, 128);
    sub(reg_a1, reg_a1, 128);
    sub(reg_a2, reg_a2, 128);
    sub(reg_a3, reg_a3, 128);
    ldrsb(reg_b0, ptr(reg_w0));
    ldrsb(reg_b1, ptr(reg_w1));
    ldrsb(reg_b2, ptr(reg_w2));
    ldrsb(reg_b3, ptr(reg_w3));

    add(reg_src0, reg_src0, 1);
    add(reg_src1, reg_src1, 1);
    add(reg_src2, reg_src2, 1);
    add(reg_src3, reg_src3, 1);
    add(reg_w0, reg_w0, 1);
    add(reg_w1, reg_w1, 1);
    add(reg_w2, reg_w2, 1);
    add(reg_w3, reg_w3, 1);

    ldr(reg_tmp, ptr(reg_c0));
    madd(reg_tmp, reg_a0, reg_b0, reg_tmp);
    str(reg_tmp, ptr(reg_c0));
    ldr(reg_tmp, ptr(reg_c0, 4));
    madd(reg_tmp, reg_a0, reg_b1, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 4));
    ldr(reg_tmp, ptr(reg_c0, 8));
    madd(reg_tmp, reg_a0, reg_b2, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 8));
    ldr(reg_tmp, ptr(reg_c0, 12));
    madd(reg_tmp, reg_a0, reg_b3, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 12));

    ldr(reg_tmp, ptr(reg_c1));
    madd(reg_tmp, reg_a1, reg_b0, reg_tmp);
    str(reg_tmp, ptr(reg_c1));
    ldr(reg_tmp, ptr(reg_c1, 4));
    madd(reg_tmp, reg_a1, reg_b1, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 4));
    ldr(reg_tmp, ptr(reg_c1, 8));
    madd(reg_tmp, reg_a1, reg_b2, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 8));
    ldr(reg_tmp, ptr(reg_c1, 12));
    madd(reg_tmp, reg_a1, reg_b3, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 12));

    ldr(reg_tmp, ptr(reg_c2));
    madd(reg_tmp, reg_a2, reg_b0, reg_tmp);
    str(reg_tmp, ptr(reg_c2));
    ldr(reg_tmp, ptr(reg_c2, 4));
    madd(reg_tmp, reg_a2, reg_b1, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 4));
    ldr(reg_tmp, ptr(reg_c2, 8));
    madd(reg_tmp, reg_a2, reg_b2, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 8));
    ldr(reg_tmp, ptr(reg_c2, 12));
    madd(reg_tmp, reg_a2, reg_b3, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 12));

    ldr(reg_tmp, ptr(reg_c3));
    madd(reg_tmp, reg_a3, reg_b0, reg_tmp);
    str(reg_tmp, ptr(reg_c3));
    ldr(reg_tmp, ptr(reg_c3, 4));
    madd(reg_tmp, reg_a3, reg_b1, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 4));
    ldr(reg_tmp, ptr(reg_c3, 8));
    madd(reg_tmp, reg_a3, reg_b2, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 8));
    ldr(reg_tmp, ptr(reg_c3, 12));
    madd(reg_tmp, reg_a3, reg_b3, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 12));

    subs(reg_k, reg_k, 1);
    b(NE, tail_loop);

    L(done);
    postamble();
}

jit_int8_brgemm_kernel_4x4_dot_packed::jit_int8_brgemm_kernel_4x4_dot_packed() : jit_generator() {}

void jit_int8_brgemm_kernel_4x4_dot_packed::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_4x4_dot_packed::generate() {
    preamble();

    const XReg reg_srcs = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_ldc = abi_param6;
    const XReg reg_accum = abi_param7;

    const XReg reg_src0 = x10;
    const XReg reg_src1 = x11;
    const XReg reg_src2 = x12;
    const XReg reg_src3 = x13;

    const XReg reg_c0 = x18;
    const XReg reg_c1 = x19;
    const XReg reg_c2 = x20;
    const XReg reg_c3 = x21;

    const VReg16B v_src0 = VReg16B(0);
    const VReg16B v_src1 = VReg16B(1);
    const VReg16B v_src2 = VReg16B(2);
    const VReg16B v_src3 = VReg16B(3);
    const VReg16B v_w0 = VReg16B(4);
    const VReg16B v_w1 = VReg16B(5);
    const VReg16B v_w2 = VReg16B(6);
    const VReg16B v_w3 = VReg16B(7);
    const VReg16BList v_wlist(v_w0, v_w3);

    const VReg4S v_acc00 = VReg4S(16);
    const VReg4S v_acc01 = VReg4S(17);
    const VReg4S v_acc02 = VReg4S(18);
    const VReg4S v_acc03 = VReg4S(19);
    const VReg4S v_acc10 = VReg4S(20);
    const VReg4S v_acc11 = VReg4S(21);
    const VReg4S v_acc12 = VReg4S(22);
    const VReg4S v_acc13 = VReg4S(23);
    const VReg4S v_acc20 = VReg4S(24);
    const VReg4S v_acc21 = VReg4S(25);
    const VReg4S v_acc22 = VReg4S(26);
    const VReg4S v_acc23 = VReg4S(27);
    const VReg4S v_acc30 = VReg4S(28);
    const VReg4S v_acc31 = VReg4S(29);
    const VReg4S v_acc32 = VReg4S(30);
    const VReg4S v_acc33 = VReg4S(31);

    const VReg16B v_acc00b(v_acc00.getIdx());
    const VReg16B v_acc01b(v_acc01.getIdx());
    const VReg16B v_acc02b(v_acc02.getIdx());
    const VReg16B v_acc03b(v_acc03.getIdx());
    const VReg16B v_acc10b(v_acc10.getIdx());
    const VReg16B v_acc11b(v_acc11.getIdx());
    const VReg16B v_acc12b(v_acc12.getIdx());
    const VReg16B v_acc13b(v_acc13.getIdx());
    const VReg16B v_acc20b(v_acc20.getIdx());
    const VReg16B v_acc21b(v_acc21.getIdx());
    const VReg16B v_acc22b(v_acc22.getIdx());
    const VReg16B v_acc23b(v_acc23.getIdx());
    const VReg16B v_acc30b(v_acc30.getIdx());
    const VReg16B v_acc31b(v_acc31.getIdx());
    const VReg16B v_acc32b(v_acc32.getIdx());
    const VReg16B v_acc33b(v_acc33.getIdx());

    eor(v_acc00b, v_acc00b, v_acc00b);
    eor(v_acc01b, v_acc01b, v_acc01b);
    eor(v_acc02b, v_acc02b, v_acc02b);
    eor(v_acc03b, v_acc03b, v_acc03b);
    eor(v_acc10b, v_acc10b, v_acc10b);
    eor(v_acc11b, v_acc11b, v_acc11b);
    eor(v_acc12b, v_acc12b, v_acc12b);
    eor(v_acc13b, v_acc13b, v_acc13b);
    eor(v_acc20b, v_acc20b, v_acc20b);
    eor(v_acc21b, v_acc21b, v_acc21b);
    eor(v_acc22b, v_acc22b, v_acc22b);
    eor(v_acc23b, v_acc23b, v_acc23b);
    eor(v_acc30b, v_acc30b, v_acc30b);
    eor(v_acc31b, v_acc31b, v_acc31b);
    eor(v_acc32b, v_acc32b, v_acc32b);
    eor(v_acc33b, v_acc33b, v_acc33b);

    ldr(reg_src0, ptr(reg_srcs));
    ldr(reg_src1, ptr(reg_srcs, 8));
    ldr(reg_src2, ptr(reg_srcs, 16));
    ldr(reg_src3, ptr(reg_srcs, 24));

    mov(reg_c0, reg_dst);
    add(reg_c1, reg_c0, reg_ldc);
    add(reg_c2, reg_c1, reg_ldc);
    add(reg_c3, reg_c2, reg_ldc);

    Label loop32;
    Label loop16;
    Label reduce_store;
    Label tail_loop;
    Label done;

    cmp(reg_k, 32);
    b(LT, loop16);

    L(loop32);
    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(v_src2, ptr(reg_src2));
    ld1(v_src3, ptr(reg_src3));
    ld1(v_wlist, ptr(reg_wei));
    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
    add(reg_src2, reg_src2, 16);
    add(reg_src3, reg_src3, 16);
    add(reg_wei, reg_wei, 64);

    sdot(v_acc00, v_src0, v_w0);
    sdot(v_acc01, v_src0, v_w1);
    sdot(v_acc02, v_src0, v_w2);
    sdot(v_acc03, v_src0, v_w3);

    sdot(v_acc10, v_src1, v_w0);
    sdot(v_acc11, v_src1, v_w1);
    sdot(v_acc12, v_src1, v_w2);
    sdot(v_acc13, v_src1, v_w3);

    sdot(v_acc20, v_src2, v_w0);
    sdot(v_acc21, v_src2, v_w1);
    sdot(v_acc22, v_src2, v_w2);
    sdot(v_acc23, v_src2, v_w3);

    sdot(v_acc30, v_src3, v_w0);
    sdot(v_acc31, v_src3, v_w1);
    sdot(v_acc32, v_src3, v_w2);
    sdot(v_acc33, v_src3, v_w3);

    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(v_src2, ptr(reg_src2));
    ld1(v_src3, ptr(reg_src3));
    ld1(v_wlist, ptr(reg_wei));
    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
    add(reg_src2, reg_src2, 16);
    add(reg_src3, reg_src3, 16);
    add(reg_wei, reg_wei, 64);

    sdot(v_acc00, v_src0, v_w0);
    sdot(v_acc01, v_src0, v_w1);
    sdot(v_acc02, v_src0, v_w2);
    sdot(v_acc03, v_src0, v_w3);

    sdot(v_acc10, v_src1, v_w0);
    sdot(v_acc11, v_src1, v_w1);
    sdot(v_acc12, v_src1, v_w2);
    sdot(v_acc13, v_src1, v_w3);

    sdot(v_acc20, v_src2, v_w0);
    sdot(v_acc21, v_src2, v_w1);
    sdot(v_acc22, v_src2, v_w2);
    sdot(v_acc23, v_src2, v_w3);

    sdot(v_acc30, v_src3, v_w0);
    sdot(v_acc31, v_src3, v_w1);
    sdot(v_acc32, v_src3, v_w2);
    sdot(v_acc33, v_src3, v_w3);

    sub(reg_k, reg_k, 32);
    cmp(reg_k, 32);
    b(GE, loop32);

    L(loop16);
    cmp(reg_k, 16);
    b(LT, reduce_store);
    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(v_src2, ptr(reg_src2));
    ld1(v_src3, ptr(reg_src3));
    ld1(v_wlist, ptr(reg_wei));
    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
    add(reg_src2, reg_src2, 16);
    add(reg_src3, reg_src3, 16);
    add(reg_wei, reg_wei, 64);

    sdot(v_acc00, v_src0, v_w0);
    sdot(v_acc01, v_src0, v_w1);
    sdot(v_acc02, v_src0, v_w2);
    sdot(v_acc03, v_src0, v_w3);

    sdot(v_acc10, v_src1, v_w0);
    sdot(v_acc11, v_src1, v_w1);
    sdot(v_acc12, v_src1, v_w2);
    sdot(v_acc13, v_src1, v_w3);

    sdot(v_acc20, v_src2, v_w0);
    sdot(v_acc21, v_src2, v_w1);
    sdot(v_acc22, v_src2, v_w2);
    sdot(v_acc23, v_src2, v_w3);

    sdot(v_acc30, v_src3, v_w0);
    sdot(v_acc31, v_src3, v_w1);
    sdot(v_acc32, v_src3, v_w2);
    sdot(v_acc33, v_src3, v_w3);

    sub(reg_k, reg_k, 16);

    L(reduce_store);
    const VReg v_tmp(0);
    const SReg s_tmp(v_tmp.getIdx());
    const WReg reg_tmp = w9;
    const WReg reg_c_acc = w0;
    const WReg reg_accum_w(reg_accum.getIdx());

    addv(s_tmp, v_acc00);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0));
    addv(s_tmp, v_acc01);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 4));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 4));
    addv(s_tmp, v_acc02);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 8));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 8));
    addv(s_tmp, v_acc03);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 12));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 12));

    addv(s_tmp, v_acc10);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1));
    addv(s_tmp, v_acc11);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 4));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 4));
    addv(s_tmp, v_acc12);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 8));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 8));
    addv(s_tmp, v_acc13);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 12));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 12));

    addv(s_tmp, v_acc20);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c2));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c2));
    addv(s_tmp, v_acc21);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c2, 4));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 4));
    addv(s_tmp, v_acc22);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c2, 8));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 8));
    addv(s_tmp, v_acc23);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c2, 12));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 12));

    addv(s_tmp, v_acc30);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c3));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c3));
    addv(s_tmp, v_acc31);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c3, 4));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 4));
    addv(s_tmp, v_acc32);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c3, 8));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 8));
    addv(s_tmp, v_acc33);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c3, 12));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 12));

    cbz(reg_k, done);

    const WReg reg_a0 = w22;
    const WReg reg_a1 = w23;
    const WReg reg_a2 = w24;
    const WReg reg_a3 = w25;
    const WReg reg_b0 = w26;
    const WReg reg_b1 = w27;
    const WReg reg_b2 = w28;
    const WReg reg_b3 = w8;

    L(tail_loop);
    ldrsb(reg_a0, ptr(reg_src0));
    ldrsb(reg_a1, ptr(reg_src1));
    ldrsb(reg_a2, ptr(reg_src2));
    ldrsb(reg_a3, ptr(reg_src3));
    ldrsb(reg_b0, ptr(reg_wei));
    ldrsb(reg_b1, ptr(reg_wei, 1));
    ldrsb(reg_b2, ptr(reg_wei, 2));
    ldrsb(reg_b3, ptr(reg_wei, 3));

    add(reg_src0, reg_src0, 1);
    add(reg_src1, reg_src1, 1);
    add(reg_src2, reg_src2, 1);
    add(reg_src3, reg_src3, 1);
    add(reg_wei, reg_wei, 4);

    ldr(reg_tmp, ptr(reg_c0));
    madd(reg_tmp, reg_a0, reg_b0, reg_tmp);
    str(reg_tmp, ptr(reg_c0));
    ldr(reg_tmp, ptr(reg_c0, 4));
    madd(reg_tmp, reg_a0, reg_b1, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 4));
    ldr(reg_tmp, ptr(reg_c0, 8));
    madd(reg_tmp, reg_a0, reg_b2, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 8));
    ldr(reg_tmp, ptr(reg_c0, 12));
    madd(reg_tmp, reg_a0, reg_b3, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 12));

    ldr(reg_tmp, ptr(reg_c1));
    madd(reg_tmp, reg_a1, reg_b0, reg_tmp);
    str(reg_tmp, ptr(reg_c1));
    ldr(reg_tmp, ptr(reg_c1, 4));
    madd(reg_tmp, reg_a1, reg_b1, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 4));
    ldr(reg_tmp, ptr(reg_c1, 8));
    madd(reg_tmp, reg_a1, reg_b2, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 8));
    ldr(reg_tmp, ptr(reg_c1, 12));
    madd(reg_tmp, reg_a1, reg_b3, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 12));

    ldr(reg_tmp, ptr(reg_c2));
    madd(reg_tmp, reg_a2, reg_b0, reg_tmp);
    str(reg_tmp, ptr(reg_c2));
    ldr(reg_tmp, ptr(reg_c2, 4));
    madd(reg_tmp, reg_a2, reg_b1, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 4));
    ldr(reg_tmp, ptr(reg_c2, 8));
    madd(reg_tmp, reg_a2, reg_b2, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 8));
    ldr(reg_tmp, ptr(reg_c2, 12));
    madd(reg_tmp, reg_a2, reg_b3, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 12));

    ldr(reg_tmp, ptr(reg_c3));
    madd(reg_tmp, reg_a3, reg_b0, reg_tmp);
    str(reg_tmp, ptr(reg_c3));
    ldr(reg_tmp, ptr(reg_c3, 4));
    madd(reg_tmp, reg_a3, reg_b1, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 4));
    ldr(reg_tmp, ptr(reg_c3, 8));
    madd(reg_tmp, reg_a3, reg_b2, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 8));
    ldr(reg_tmp, ptr(reg_c3, 12));
    madd(reg_tmp, reg_a3, reg_b3, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 12));

    subs(reg_k, reg_k, 1);
    b(NE, tail_loop);

    L(done);
    postamble();
}

jit_int8_brgemm_kernel_4x4_smmla_packed::jit_int8_brgemm_kernel_4x4_smmla_packed() : jit_generator() {}

void jit_int8_brgemm_kernel_4x4_smmla_packed::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_4x4_smmla_packed::generate() {
    preamble();

    const XReg reg_srcs = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_ldc = abi_param6;
    const XReg reg_accum = abi_param7;

    const XReg reg_src0 = x10;
    const XReg reg_src1 = x11;
    const XReg reg_src2 = x12;
    const XReg reg_src3 = x13;

    const XReg reg_c0 = x18;
    const XReg reg_c1 = x19;
    const XReg reg_c2 = x20;
    const XReg reg_c3 = x21;

    const VReg v_a01 = VReg(0);
    const VReg v_a23 = VReg(1);
    const VReg v_tmp0 = VReg(2);
    const VReg v_tmp1 = VReg(3);
    const VReg v_w01 = VReg(4);
    const VReg v_w23 = VReg(5);

    const VReg v_acc00 = VReg(16);
    const VReg v_acc01 = VReg(17);
    const VReg v_acc10 = VReg(18);
    const VReg v_acc11 = VReg(19);

    eor(v_acc00.b, v_acc00.b, v_acc00.b);
    eor(v_acc01.b, v_acc01.b, v_acc01.b);
    eor(v_acc10.b, v_acc10.b, v_acc10.b);
    eor(v_acc11.b, v_acc11.b, v_acc11.b);

    ldr(reg_src0, ptr(reg_srcs));
    ldr(reg_src1, ptr(reg_srcs, 8));
    ldr(reg_src2, ptr(reg_srcs, 16));
    ldr(reg_src3, ptr(reg_srcs, 24));

    mov(reg_c0, reg_dst);
    add(reg_c1, reg_c0, reg_ldc);
    add(reg_c2, reg_c1, reg_ldc);
    add(reg_c3, reg_c2, reg_ldc);

    auto emit_smmla = [&](const VReg4S& vd, const VReg16B& vn, const VReg16B& vm) {
        const uint32_t base = 0x4e80a400;
        const uint32_t code = base | static_cast<uint32_t>(vd.getIdx()) |
                              (static_cast<uint32_t>(vn.getIdx()) << 5) |
                              (static_cast<uint32_t>(vm.getIdx()) << 16);
        dd(code);
    };

    Label loop16;
    Label loop8;
    Label done;

    cbz(reg_k, done);

    cmp(reg_k, 16);
    b(LT, loop8);

    L(loop16);
    ldr(DReg(v_a01.getIdx()), ptr(reg_src0));
    ldr(DReg(v_tmp0.getIdx()), ptr(reg_src1));
    ins(v_a01.d[1], v_tmp0.d[0]);
    ldr(DReg(v_a23.getIdx()), ptr(reg_src2));
    ldr(DReg(v_tmp1.getIdx()), ptr(reg_src3));
    ins(v_a23.d[1], v_tmp1.d[0]);

    ldr(QReg(v_w01.getIdx()), ptr(reg_wei));
    ldr(QReg(v_w23.getIdx()), ptr(reg_wei, 16));

    emit_smmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_smmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_smmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_smmla(v_acc11.s, v_a23.b, v_w23.b);

    add(reg_src0, reg_src0, 8);
    add(reg_src1, reg_src1, 8);
    add(reg_src2, reg_src2, 8);
    add(reg_src3, reg_src3, 8);
    add(reg_wei, reg_wei, 32);

    ldr(DReg(v_a01.getIdx()), ptr(reg_src0));
    ldr(DReg(v_tmp0.getIdx()), ptr(reg_src1));
    ins(v_a01.d[1], v_tmp0.d[0]);
    ldr(DReg(v_a23.getIdx()), ptr(reg_src2));
    ldr(DReg(v_tmp1.getIdx()), ptr(reg_src3));
    ins(v_a23.d[1], v_tmp1.d[0]);

    ldr(QReg(v_w01.getIdx()), ptr(reg_wei));
    ldr(QReg(v_w23.getIdx()), ptr(reg_wei, 16));

    emit_smmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_smmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_smmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_smmla(v_acc11.s, v_a23.b, v_w23.b);

    add(reg_src0, reg_src0, 8);
    add(reg_src1, reg_src1, 8);
    add(reg_src2, reg_src2, 8);
    add(reg_src3, reg_src3, 8);
    add(reg_wei, reg_wei, 32);

    subs(reg_k, reg_k, 16);
    cmp(reg_k, 16);
    b(GE, loop16);

    L(loop8);
    cbz(reg_k, done);
    ldr(DReg(v_a01.getIdx()), ptr(reg_src0));
    ldr(DReg(v_tmp0.getIdx()), ptr(reg_src1));
    ins(v_a01.d[1], v_tmp0.d[0]);
    ldr(DReg(v_a23.getIdx()), ptr(reg_src2));
    ldr(DReg(v_tmp1.getIdx()), ptr(reg_src3));
    ins(v_a23.d[1], v_tmp1.d[0]);

    ldr(QReg(v_w01.getIdx()), ptr(reg_wei));
    ldr(QReg(v_w23.getIdx()), ptr(reg_wei, 16));

    emit_smmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_smmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_smmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_smmla(v_acc11.s, v_a23.b, v_w23.b);

    add(reg_src0, reg_src0, 8);
    add(reg_src1, reg_src1, 8);
    add(reg_src2, reg_src2, 8);
    add(reg_src3, reg_src3, 8);
    add(reg_wei, reg_wei, 32);

    subs(reg_k, reg_k, 8);
    // reg_k is expected to hit zero here

    const WReg reg_tmp = w9;
    const WReg reg_c_acc = w0;
    Label store_no_accum;
    Label store_accum;
    Label store_done;

    cbz(reg_accum, store_no_accum);

    L(store_accum);
    umov(reg_tmp, v_acc00.s[0]);
    ldr(reg_c_acc, ptr(reg_c0));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0));
    umov(reg_tmp, v_acc00.s[1]);
    ldr(reg_c_acc, ptr(reg_c0, 4));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0, 4));
    umov(reg_tmp, v_acc01.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 8));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0, 8));
    umov(reg_tmp, v_acc01.s[1]);
    ldr(reg_c_acc, ptr(reg_c0, 12));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0, 12));

    umov(reg_tmp, v_acc00.s[2]);
    ldr(reg_c_acc, ptr(reg_c1));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1));
    umov(reg_tmp, v_acc00.s[3]);
    ldr(reg_c_acc, ptr(reg_c1, 4));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1, 4));
    umov(reg_tmp, v_acc01.s[2]);
    ldr(reg_c_acc, ptr(reg_c1, 8));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1, 8));
    umov(reg_tmp, v_acc01.s[3]);
    ldr(reg_c_acc, ptr(reg_c1, 12));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1, 12));

    umov(reg_tmp, v_acc10.s[0]);
    ldr(reg_c_acc, ptr(reg_c2));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2));
    umov(reg_tmp, v_acc10.s[1]);
    ldr(reg_c_acc, ptr(reg_c2, 4));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2, 4));
    umov(reg_tmp, v_acc11.s[0]);
    ldr(reg_c_acc, ptr(reg_c2, 8));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2, 8));
    umov(reg_tmp, v_acc11.s[1]);
    ldr(reg_c_acc, ptr(reg_c2, 12));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2, 12));

    umov(reg_tmp, v_acc10.s[2]);
    ldr(reg_c_acc, ptr(reg_c3));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3));
    umov(reg_tmp, v_acc10.s[3]);
    ldr(reg_c_acc, ptr(reg_c3, 4));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3, 4));
    umov(reg_tmp, v_acc11.s[2]);
    ldr(reg_c_acc, ptr(reg_c3, 8));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3, 8));
    umov(reg_tmp, v_acc11.s[3]);
    ldr(reg_c_acc, ptr(reg_c3, 12));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3, 12));
    b(store_done);

    L(store_no_accum);
    umov(reg_tmp, v_acc00.s[0]);
    str(reg_tmp, ptr(reg_c0));
    umov(reg_tmp, v_acc00.s[1]);
    str(reg_tmp, ptr(reg_c0, 4));
    umov(reg_tmp, v_acc01.s[0]);
    str(reg_tmp, ptr(reg_c0, 8));
    umov(reg_tmp, v_acc01.s[1]);
    str(reg_tmp, ptr(reg_c0, 12));

    umov(reg_tmp, v_acc00.s[2]);
    str(reg_tmp, ptr(reg_c1));
    umov(reg_tmp, v_acc00.s[3]);
    str(reg_tmp, ptr(reg_c1, 4));
    umov(reg_tmp, v_acc01.s[2]);
    str(reg_tmp, ptr(reg_c1, 8));
    umov(reg_tmp, v_acc01.s[3]);
    str(reg_tmp, ptr(reg_c1, 12));

    umov(reg_tmp, v_acc10.s[0]);
    str(reg_tmp, ptr(reg_c2));
    umov(reg_tmp, v_acc10.s[1]);
    str(reg_tmp, ptr(reg_c2, 4));
    umov(reg_tmp, v_acc11.s[0]);
    str(reg_tmp, ptr(reg_c2, 8));
    umov(reg_tmp, v_acc11.s[1]);
    str(reg_tmp, ptr(reg_c2, 12));

    umov(reg_tmp, v_acc10.s[2]);
    str(reg_tmp, ptr(reg_c3));
    umov(reg_tmp, v_acc10.s[3]);
    str(reg_tmp, ptr(reg_c3, 4));
    umov(reg_tmp, v_acc11.s[2]);
    str(reg_tmp, ptr(reg_c3, 8));
    umov(reg_tmp, v_acc11.s[3]);
    str(reg_tmp, ptr(reg_c3, 12));

    L(store_done);

    L(done);
    postamble();
}

jit_int8_brgemm_kernel_4x8_smmla_packed::jit_int8_brgemm_kernel_4x8_smmla_packed() : jit_generator() {}

void jit_int8_brgemm_kernel_4x8_smmla_packed::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_4x8_smmla_packed::generate() {
    preamble();

    const XReg reg_srcs = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_ldc = abi_param6;
    const XReg reg_accum = abi_param7;

    const XReg reg_src0 = x10;
    const XReg reg_src1 = x11;
    const XReg reg_src2 = x12;
    const XReg reg_src3 = x13;

    const XReg reg_c0 = x18;
    const XReg reg_c1 = x19;
    const XReg reg_c2 = x20;
    const XReg reg_c3 = x21;

    const VReg v_a01 = VReg(0);
    const VReg v_a23 = VReg(1);
    const VReg v_tmp0 = VReg(2);
    const VReg v_tmp1 = VReg(3);
    const VReg v_w01 = VReg(4);
    const VReg v_w23 = VReg(5);
    const VReg v_w45 = VReg(6);
    const VReg v_w67 = VReg(7);

    const VReg v_acc00 = VReg(16);
    const VReg v_acc01 = VReg(17);
    const VReg v_acc02 = VReg(18);
    const VReg v_acc03 = VReg(19);
    const VReg v_acc10 = VReg(20);
    const VReg v_acc11 = VReg(21);
    const VReg v_acc12 = VReg(22);
    const VReg v_acc13 = VReg(23);

    eor(v_acc00.b, v_acc00.b, v_acc00.b);
    eor(v_acc01.b, v_acc01.b, v_acc01.b);
    eor(v_acc02.b, v_acc02.b, v_acc02.b);
    eor(v_acc03.b, v_acc03.b, v_acc03.b);
    eor(v_acc10.b, v_acc10.b, v_acc10.b);
    eor(v_acc11.b, v_acc11.b, v_acc11.b);
    eor(v_acc12.b, v_acc12.b, v_acc12.b);
    eor(v_acc13.b, v_acc13.b, v_acc13.b);

    ldr(reg_src0, ptr(reg_srcs));
    ldr(reg_src1, ptr(reg_srcs, 8));
    ldr(reg_src2, ptr(reg_srcs, 16));
    ldr(reg_src3, ptr(reg_srcs, 24));

    mov(reg_c0, reg_dst);
    add(reg_c1, reg_c0, reg_ldc);
    add(reg_c2, reg_c1, reg_ldc);
    add(reg_c3, reg_c2, reg_ldc);

    auto emit_smmla = [&](const VReg4S& vd, const VReg16B& vn, const VReg16B& vm) {
        const uint32_t base = 0x4e80a400;
        const uint32_t code = base | static_cast<uint32_t>(vd.getIdx()) |
                              (static_cast<uint32_t>(vn.getIdx()) << 5) |
                              (static_cast<uint32_t>(vm.getIdx()) << 16);
        dd(code);
    };

    Label loop16;
    Label loop8;
    Label done;

    cbz(reg_k, done);

    cmp(reg_k, 16);
    b(LT, loop8);

    L(loop16);
    ldr(DReg(v_a01.getIdx()), ptr(reg_src0));
    ldr(DReg(v_tmp0.getIdx()), ptr(reg_src1));
    ins(v_a01.d[1], v_tmp0.d[0]);
    ldr(DReg(v_a23.getIdx()), ptr(reg_src2));
    ldr(DReg(v_tmp1.getIdx()), ptr(reg_src3));
    ins(v_a23.d[1], v_tmp1.d[0]);

    ldr(QReg(v_w01.getIdx()), ptr(reg_wei));
    ldr(QReg(v_w23.getIdx()), ptr(reg_wei, 16));
    ldr(QReg(v_w45.getIdx()), ptr(reg_wei, 32));
    ldr(QReg(v_w67.getIdx()), ptr(reg_wei, 48));

    emit_smmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_smmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_smmla(v_acc02.s, v_a01.b, v_w45.b);
    emit_smmla(v_acc03.s, v_a01.b, v_w67.b);
    emit_smmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_smmla(v_acc11.s, v_a23.b, v_w23.b);
    emit_smmla(v_acc12.s, v_a23.b, v_w45.b);
    emit_smmla(v_acc13.s, v_a23.b, v_w67.b);

    add(reg_src0, reg_src0, 8);
    add(reg_src1, reg_src1, 8);
    add(reg_src2, reg_src2, 8);
    add(reg_src3, reg_src3, 8);
    add(reg_wei, reg_wei, 64);

    ldr(DReg(v_a01.getIdx()), ptr(reg_src0));
    ldr(DReg(v_tmp0.getIdx()), ptr(reg_src1));
    ins(v_a01.d[1], v_tmp0.d[0]);
    ldr(DReg(v_a23.getIdx()), ptr(reg_src2));
    ldr(DReg(v_tmp1.getIdx()), ptr(reg_src3));
    ins(v_a23.d[1], v_tmp1.d[0]);

    ldr(QReg(v_w01.getIdx()), ptr(reg_wei));
    ldr(QReg(v_w23.getIdx()), ptr(reg_wei, 16));
    ldr(QReg(v_w45.getIdx()), ptr(reg_wei, 32));
    ldr(QReg(v_w67.getIdx()), ptr(reg_wei, 48));

    emit_smmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_smmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_smmla(v_acc02.s, v_a01.b, v_w45.b);
    emit_smmla(v_acc03.s, v_a01.b, v_w67.b);
    emit_smmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_smmla(v_acc11.s, v_a23.b, v_w23.b);
    emit_smmla(v_acc12.s, v_a23.b, v_w45.b);
    emit_smmla(v_acc13.s, v_a23.b, v_w67.b);

    add(reg_src0, reg_src0, 8);
    add(reg_src1, reg_src1, 8);
    add(reg_src2, reg_src2, 8);
    add(reg_src3, reg_src3, 8);
    add(reg_wei, reg_wei, 64);

    subs(reg_k, reg_k, 16);
    cmp(reg_k, 16);
    b(GE, loop16);

    L(loop8);
    cbz(reg_k, done);
    ldr(DReg(v_a01.getIdx()), ptr(reg_src0));
    ldr(DReg(v_tmp0.getIdx()), ptr(reg_src1));
    ins(v_a01.d[1], v_tmp0.d[0]);
    ldr(DReg(v_a23.getIdx()), ptr(reg_src2));
    ldr(DReg(v_tmp1.getIdx()), ptr(reg_src3));
    ins(v_a23.d[1], v_tmp1.d[0]);

    ldr(QReg(v_w01.getIdx()), ptr(reg_wei));
    ldr(QReg(v_w23.getIdx()), ptr(reg_wei, 16));
    ldr(QReg(v_w45.getIdx()), ptr(reg_wei, 32));
    ldr(QReg(v_w67.getIdx()), ptr(reg_wei, 48));

    emit_smmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_smmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_smmla(v_acc02.s, v_a01.b, v_w45.b);
    emit_smmla(v_acc03.s, v_a01.b, v_w67.b);
    emit_smmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_smmla(v_acc11.s, v_a23.b, v_w23.b);
    emit_smmla(v_acc12.s, v_a23.b, v_w45.b);
    emit_smmla(v_acc13.s, v_a23.b, v_w67.b);

    add(reg_src0, reg_src0, 8);
    add(reg_src1, reg_src1, 8);
    add(reg_src2, reg_src2, 8);
    add(reg_src3, reg_src3, 8);
    add(reg_wei, reg_wei, 64);

    subs(reg_k, reg_k, 8);
    // reg_k is expected to hit zero here

    const WReg reg_tmp = w9;
    const WReg reg_c_acc = w0;
    Label store_no_accum;
    Label store_accum;
    Label store_done;

    cbz(reg_accum, store_no_accum);

    L(store_accum);
    umov(reg_tmp, v_acc00.s[0]);
    ldr(reg_c_acc, ptr(reg_c0));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0));
    umov(reg_tmp, v_acc00.s[1]);
    ldr(reg_c_acc, ptr(reg_c0, 4));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0, 4));
    umov(reg_tmp, v_acc01.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 8));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0, 8));
    umov(reg_tmp, v_acc01.s[1]);
    ldr(reg_c_acc, ptr(reg_c0, 12));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0, 12));
    umov(reg_tmp, v_acc02.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 16));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0, 16));
    umov(reg_tmp, v_acc02.s[1]);
    ldr(reg_c_acc, ptr(reg_c0, 20));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0, 20));
    umov(reg_tmp, v_acc03.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 24));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0, 24));
    umov(reg_tmp, v_acc03.s[1]);
    ldr(reg_c_acc, ptr(reg_c0, 28));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0, 28));

    umov(reg_tmp, v_acc00.s[2]);
    ldr(reg_c_acc, ptr(reg_c1));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1));
    umov(reg_tmp, v_acc00.s[3]);
    ldr(reg_c_acc, ptr(reg_c1, 4));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1, 4));
    umov(reg_tmp, v_acc01.s[2]);
    ldr(reg_c_acc, ptr(reg_c1, 8));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1, 8));
    umov(reg_tmp, v_acc01.s[3]);
    ldr(reg_c_acc, ptr(reg_c1, 12));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1, 12));
    umov(reg_tmp, v_acc02.s[2]);
    ldr(reg_c_acc, ptr(reg_c1, 16));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1, 16));
    umov(reg_tmp, v_acc02.s[3]);
    ldr(reg_c_acc, ptr(reg_c1, 20));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1, 20));
    umov(reg_tmp, v_acc03.s[2]);
    ldr(reg_c_acc, ptr(reg_c1, 24));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1, 24));
    umov(reg_tmp, v_acc03.s[3]);
    ldr(reg_c_acc, ptr(reg_c1, 28));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1, 28));

    umov(reg_tmp, v_acc10.s[0]);
    ldr(reg_c_acc, ptr(reg_c2));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2));
    umov(reg_tmp, v_acc10.s[1]);
    ldr(reg_c_acc, ptr(reg_c2, 4));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2, 4));
    umov(reg_tmp, v_acc11.s[0]);
    ldr(reg_c_acc, ptr(reg_c2, 8));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2, 8));
    umov(reg_tmp, v_acc11.s[1]);
    ldr(reg_c_acc, ptr(reg_c2, 12));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2, 12));
    umov(reg_tmp, v_acc12.s[0]);
    ldr(reg_c_acc, ptr(reg_c2, 16));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2, 16));
    umov(reg_tmp, v_acc12.s[1]);
    ldr(reg_c_acc, ptr(reg_c2, 20));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2, 20));
    umov(reg_tmp, v_acc13.s[0]);
    ldr(reg_c_acc, ptr(reg_c2, 24));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2, 24));
    umov(reg_tmp, v_acc13.s[1]);
    ldr(reg_c_acc, ptr(reg_c2, 28));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2, 28));

    umov(reg_tmp, v_acc10.s[2]);
    ldr(reg_c_acc, ptr(reg_c3));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3));
    umov(reg_tmp, v_acc10.s[3]);
    ldr(reg_c_acc, ptr(reg_c3, 4));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3, 4));
    umov(reg_tmp, v_acc11.s[2]);
    ldr(reg_c_acc, ptr(reg_c3, 8));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3, 8));
    umov(reg_tmp, v_acc11.s[3]);
    ldr(reg_c_acc, ptr(reg_c3, 12));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3, 12));
    umov(reg_tmp, v_acc12.s[2]);
    ldr(reg_c_acc, ptr(reg_c3, 16));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3, 16));
    umov(reg_tmp, v_acc12.s[3]);
    ldr(reg_c_acc, ptr(reg_c3, 20));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3, 20));
    umov(reg_tmp, v_acc13.s[2]);
    ldr(reg_c_acc, ptr(reg_c3, 24));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3, 24));
    umov(reg_tmp, v_acc13.s[3]);
    ldr(reg_c_acc, ptr(reg_c3, 28));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3, 28));
    b(store_done);

    L(store_no_accum);
    umov(reg_tmp, v_acc00.s[0]);
    str(reg_tmp, ptr(reg_c0));
    umov(reg_tmp, v_acc00.s[1]);
    str(reg_tmp, ptr(reg_c0, 4));
    umov(reg_tmp, v_acc01.s[0]);
    str(reg_tmp, ptr(reg_c0, 8));
    umov(reg_tmp, v_acc01.s[1]);
    str(reg_tmp, ptr(reg_c0, 12));
    umov(reg_tmp, v_acc02.s[0]);
    str(reg_tmp, ptr(reg_c0, 16));
    umov(reg_tmp, v_acc02.s[1]);
    str(reg_tmp, ptr(reg_c0, 20));
    umov(reg_tmp, v_acc03.s[0]);
    str(reg_tmp, ptr(reg_c0, 24));
    umov(reg_tmp, v_acc03.s[1]);
    str(reg_tmp, ptr(reg_c0, 28));

    umov(reg_tmp, v_acc00.s[2]);
    str(reg_tmp, ptr(reg_c1));
    umov(reg_tmp, v_acc00.s[3]);
    str(reg_tmp, ptr(reg_c1, 4));
    umov(reg_tmp, v_acc01.s[2]);
    str(reg_tmp, ptr(reg_c1, 8));
    umov(reg_tmp, v_acc01.s[3]);
    str(reg_tmp, ptr(reg_c1, 12));
    umov(reg_tmp, v_acc02.s[2]);
    str(reg_tmp, ptr(reg_c1, 16));
    umov(reg_tmp, v_acc02.s[3]);
    str(reg_tmp, ptr(reg_c1, 20));
    umov(reg_tmp, v_acc03.s[2]);
    str(reg_tmp, ptr(reg_c1, 24));
    umov(reg_tmp, v_acc03.s[3]);
    str(reg_tmp, ptr(reg_c1, 28));

    umov(reg_tmp, v_acc10.s[0]);
    str(reg_tmp, ptr(reg_c2));
    umov(reg_tmp, v_acc10.s[1]);
    str(reg_tmp, ptr(reg_c2, 4));
    umov(reg_tmp, v_acc11.s[0]);
    str(reg_tmp, ptr(reg_c2, 8));
    umov(reg_tmp, v_acc11.s[1]);
    str(reg_tmp, ptr(reg_c2, 12));
    umov(reg_tmp, v_acc12.s[0]);
    str(reg_tmp, ptr(reg_c2, 16));
    umov(reg_tmp, v_acc12.s[1]);
    str(reg_tmp, ptr(reg_c2, 20));
    umov(reg_tmp, v_acc13.s[0]);
    str(reg_tmp, ptr(reg_c2, 24));
    umov(reg_tmp, v_acc13.s[1]);
    str(reg_tmp, ptr(reg_c2, 28));

    umov(reg_tmp, v_acc10.s[2]);
    str(reg_tmp, ptr(reg_c3));
    umov(reg_tmp, v_acc10.s[3]);
    str(reg_tmp, ptr(reg_c3, 4));
    umov(reg_tmp, v_acc11.s[2]);
    str(reg_tmp, ptr(reg_c3, 8));
    umov(reg_tmp, v_acc11.s[3]);
    str(reg_tmp, ptr(reg_c3, 12));
    umov(reg_tmp, v_acc12.s[2]);
    str(reg_tmp, ptr(reg_c3, 16));
    umov(reg_tmp, v_acc12.s[3]);
    str(reg_tmp, ptr(reg_c3, 20));
    umov(reg_tmp, v_acc13.s[2]);
    str(reg_tmp, ptr(reg_c3, 24));
    umov(reg_tmp, v_acc13.s[3]);
    str(reg_tmp, ptr(reg_c3, 28));

    L(store_done);

    L(done);
    postamble();
}

jit_int8_brgemm_kernel_4x16_smmla_packed::jit_int8_brgemm_kernel_4x16_smmla_packed() : jit_generator() {}

void jit_int8_brgemm_kernel_4x16_smmla_packed::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_4x16_smmla_packed::generate() {
    preamble();

    const XReg reg_srcs = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_ldc = abi_param6;
    const XReg reg_accum = abi_param7;

    const XReg reg_src0 = x10;
    const XReg reg_src1 = x11;
    const XReg reg_src2 = x12;
    const XReg reg_src3 = x13;

    const XReg reg_c0 = x18;
    const XReg reg_c1 = x19;
    const XReg reg_c2 = x20;
    const XReg reg_c3 = x21;

    const VReg v_a01 = VReg(0);
    const VReg v_a23 = VReg(1);
    const VReg v_tmp0 = VReg(2);
    const VReg v_tmp1 = VReg(3);
    const VReg v_w01 = VReg(4);
    const VReg v_w23 = VReg(5);
    const VReg v_w45 = VReg(6);
    const VReg v_w67 = VReg(7);
    const VReg v_w89 = VReg(8);
    const VReg v_wab = VReg(9);
    const VReg v_wcd = VReg(10);
    const VReg v_wef = VReg(11);

    const VReg v_acc00 = VReg(16);
    const VReg v_acc01 = VReg(17);
    const VReg v_acc02 = VReg(18);
    const VReg v_acc03 = VReg(19);
    const VReg v_acc04 = VReg(20);
    const VReg v_acc05 = VReg(21);
    const VReg v_acc06 = VReg(22);
    const VReg v_acc07 = VReg(23);
    const VReg v_acc10 = VReg(24);
    const VReg v_acc11 = VReg(25);
    const VReg v_acc12 = VReg(26);
    const VReg v_acc13 = VReg(27);
    const VReg v_acc14 = VReg(28);
    const VReg v_acc15 = VReg(29);
    const VReg v_acc16 = VReg(30);
    const VReg v_acc17 = VReg(31);

    eor(v_acc00.b, v_acc00.b, v_acc00.b);
    eor(v_acc01.b, v_acc01.b, v_acc01.b);
    eor(v_acc02.b, v_acc02.b, v_acc02.b);
    eor(v_acc03.b, v_acc03.b, v_acc03.b);
    eor(v_acc04.b, v_acc04.b, v_acc04.b);
    eor(v_acc05.b, v_acc05.b, v_acc05.b);
    eor(v_acc06.b, v_acc06.b, v_acc06.b);
    eor(v_acc07.b, v_acc07.b, v_acc07.b);
    eor(v_acc10.b, v_acc10.b, v_acc10.b);
    eor(v_acc11.b, v_acc11.b, v_acc11.b);
    eor(v_acc12.b, v_acc12.b, v_acc12.b);
    eor(v_acc13.b, v_acc13.b, v_acc13.b);
    eor(v_acc14.b, v_acc14.b, v_acc14.b);
    eor(v_acc15.b, v_acc15.b, v_acc15.b);
    eor(v_acc16.b, v_acc16.b, v_acc16.b);
    eor(v_acc17.b, v_acc17.b, v_acc17.b);

    ldr(reg_src0, ptr(reg_srcs));
    ldr(reg_src1, ptr(reg_srcs, 8));
    ldr(reg_src2, ptr(reg_srcs, 16));
    ldr(reg_src3, ptr(reg_srcs, 24));

    mov(reg_c0, reg_dst);
    add(reg_c1, reg_c0, reg_ldc);
    add(reg_c2, reg_c1, reg_ldc);
    add(reg_c3, reg_c2, reg_ldc);

    auto emit_smmla = [&](const VReg4S& vd, const VReg16B& vn, const VReg16B& vm) {
        const uint32_t base = 0x4e80a400;
        const uint32_t code = base | static_cast<uint32_t>(vd.getIdx()) |
                              (static_cast<uint32_t>(vn.getIdx()) << 5) |
                              (static_cast<uint32_t>(vm.getIdx()) << 16);
        dd(code);
    };

    Label loop16;
    Label loop8;
    Label done;

    cbz(reg_k, done);

    cmp(reg_k, 16);
    b(LT, loop8);

    L(loop16);
    ldr(DReg(v_a01.getIdx()), ptr(reg_src0));
    ldr(DReg(v_tmp0.getIdx()), ptr(reg_src1));
    ins(v_a01.d[1], v_tmp0.d[0]);
    ldr(DReg(v_a23.getIdx()), ptr(reg_src2));
    ldr(DReg(v_tmp1.getIdx()), ptr(reg_src3));
    ins(v_a23.d[1], v_tmp1.d[0]);

    ldr(QReg(v_w01.getIdx()), ptr(reg_wei));
    ldr(QReg(v_w23.getIdx()), ptr(reg_wei, 16));
    ldr(QReg(v_w45.getIdx()), ptr(reg_wei, 32));
    ldr(QReg(v_w67.getIdx()), ptr(reg_wei, 48));
    ldr(QReg(v_w89.getIdx()), ptr(reg_wei, 64));
    ldr(QReg(v_wab.getIdx()), ptr(reg_wei, 80));
    ldr(QReg(v_wcd.getIdx()), ptr(reg_wei, 96));
    ldr(QReg(v_wef.getIdx()), ptr(reg_wei, 112));

    emit_smmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_smmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_smmla(v_acc02.s, v_a01.b, v_w45.b);
    emit_smmla(v_acc03.s, v_a01.b, v_w67.b);
    emit_smmla(v_acc04.s, v_a01.b, v_w89.b);
    emit_smmla(v_acc05.s, v_a01.b, v_wab.b);
    emit_smmla(v_acc06.s, v_a01.b, v_wcd.b);
    emit_smmla(v_acc07.s, v_a01.b, v_wef.b);
    emit_smmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_smmla(v_acc11.s, v_a23.b, v_w23.b);
    emit_smmla(v_acc12.s, v_a23.b, v_w45.b);
    emit_smmla(v_acc13.s, v_a23.b, v_w67.b);
    emit_smmla(v_acc14.s, v_a23.b, v_w89.b);
    emit_smmla(v_acc15.s, v_a23.b, v_wab.b);
    emit_smmla(v_acc16.s, v_a23.b, v_wcd.b);
    emit_smmla(v_acc17.s, v_a23.b, v_wef.b);

    add(reg_src0, reg_src0, 8);
    add(reg_src1, reg_src1, 8);
    add(reg_src2, reg_src2, 8);
    add(reg_src3, reg_src3, 8);
    add(reg_wei, reg_wei, 128);

    ldr(DReg(v_a01.getIdx()), ptr(reg_src0));
    ldr(DReg(v_tmp0.getIdx()), ptr(reg_src1));
    ins(v_a01.d[1], v_tmp0.d[0]);
    ldr(DReg(v_a23.getIdx()), ptr(reg_src2));
    ldr(DReg(v_tmp1.getIdx()), ptr(reg_src3));
    ins(v_a23.d[1], v_tmp1.d[0]);

    ldr(QReg(v_w01.getIdx()), ptr(reg_wei));
    ldr(QReg(v_w23.getIdx()), ptr(reg_wei, 16));
    ldr(QReg(v_w45.getIdx()), ptr(reg_wei, 32));
    ldr(QReg(v_w67.getIdx()), ptr(reg_wei, 48));
    ldr(QReg(v_w89.getIdx()), ptr(reg_wei, 64));
    ldr(QReg(v_wab.getIdx()), ptr(reg_wei, 80));
    ldr(QReg(v_wcd.getIdx()), ptr(reg_wei, 96));
    ldr(QReg(v_wef.getIdx()), ptr(reg_wei, 112));

    emit_smmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_smmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_smmla(v_acc02.s, v_a01.b, v_w45.b);
    emit_smmla(v_acc03.s, v_a01.b, v_w67.b);
    emit_smmla(v_acc04.s, v_a01.b, v_w89.b);
    emit_smmla(v_acc05.s, v_a01.b, v_wab.b);
    emit_smmla(v_acc06.s, v_a01.b, v_wcd.b);
    emit_smmla(v_acc07.s, v_a01.b, v_wef.b);
    emit_smmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_smmla(v_acc11.s, v_a23.b, v_w23.b);
    emit_smmla(v_acc12.s, v_a23.b, v_w45.b);
    emit_smmla(v_acc13.s, v_a23.b, v_w67.b);
    emit_smmla(v_acc14.s, v_a23.b, v_w89.b);
    emit_smmla(v_acc15.s, v_a23.b, v_wab.b);
    emit_smmla(v_acc16.s, v_a23.b, v_wcd.b);
    emit_smmla(v_acc17.s, v_a23.b, v_wef.b);

    add(reg_src0, reg_src0, 8);
    add(reg_src1, reg_src1, 8);
    add(reg_src2, reg_src2, 8);
    add(reg_src3, reg_src3, 8);
    add(reg_wei, reg_wei, 128);

    subs(reg_k, reg_k, 16);
    cmp(reg_k, 16);
    b(GE, loop16);

    L(loop8);
    cbz(reg_k, done);
    ldr(DReg(v_a01.getIdx()), ptr(reg_src0));
    ldr(DReg(v_tmp0.getIdx()), ptr(reg_src1));
    ins(v_a01.d[1], v_tmp0.d[0]);
    ldr(DReg(v_a23.getIdx()), ptr(reg_src2));
    ldr(DReg(v_tmp1.getIdx()), ptr(reg_src3));
    ins(v_a23.d[1], v_tmp1.d[0]);

    ldr(QReg(v_w01.getIdx()), ptr(reg_wei));
    ldr(QReg(v_w23.getIdx()), ptr(reg_wei, 16));
    ldr(QReg(v_w45.getIdx()), ptr(reg_wei, 32));
    ldr(QReg(v_w67.getIdx()), ptr(reg_wei, 48));
    ldr(QReg(v_w89.getIdx()), ptr(reg_wei, 64));
    ldr(QReg(v_wab.getIdx()), ptr(reg_wei, 80));
    ldr(QReg(v_wcd.getIdx()), ptr(reg_wei, 96));
    ldr(QReg(v_wef.getIdx()), ptr(reg_wei, 112));

    emit_smmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_smmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_smmla(v_acc02.s, v_a01.b, v_w45.b);
    emit_smmla(v_acc03.s, v_a01.b, v_w67.b);
    emit_smmla(v_acc04.s, v_a01.b, v_w89.b);
    emit_smmla(v_acc05.s, v_a01.b, v_wab.b);
    emit_smmla(v_acc06.s, v_a01.b, v_wcd.b);
    emit_smmla(v_acc07.s, v_a01.b, v_wef.b);
    emit_smmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_smmla(v_acc11.s, v_a23.b, v_w23.b);
    emit_smmla(v_acc12.s, v_a23.b, v_w45.b);
    emit_smmla(v_acc13.s, v_a23.b, v_w67.b);
    emit_smmla(v_acc14.s, v_a23.b, v_w89.b);
    emit_smmla(v_acc15.s, v_a23.b, v_wab.b);
    emit_smmla(v_acc16.s, v_a23.b, v_wcd.b);
    emit_smmla(v_acc17.s, v_a23.b, v_wef.b);

    add(reg_src0, reg_src0, 8);
    add(reg_src1, reg_src1, 8);
    add(reg_src2, reg_src2, 8);
    add(reg_src3, reg_src3, 8);
    add(reg_wei, reg_wei, 128);

    subs(reg_k, reg_k, 8);
    // reg_k is expected to hit zero here

    Label store_no_accum;
    Label store_accum;
    Label store_done;

    auto store_acc = [&](const XReg& reg_c, int offset, const VReg& v_acc, bool upper) {
        ldr(DReg(v_tmp0.getIdx()), ptr(reg_c, offset));
        if (upper) {
            mov(v_tmp1.d[0], v_acc.d[1]);
            add(v_tmp0.s, v_tmp0.s, v_tmp1.s);
        } else {
            add(v_tmp0.s, v_tmp0.s, v_acc.s);
        }
        str(DReg(v_tmp0.getIdx()), ptr(reg_c, offset));
    };

    auto store_noacc = [&](const XReg& reg_c, int offset, const VReg& v_acc, bool upper) {
        if (upper) {
            mov(v_tmp1.d[0], v_acc.d[1]);
            str(DReg(v_tmp1.getIdx()), ptr(reg_c, offset));
        } else {
            str(DReg(v_acc.getIdx()), ptr(reg_c, offset));
        }
    };

    cbz(reg_accum, store_no_accum);

    L(store_accum);
    store_acc(reg_c0, 0, v_acc00, false);
    store_acc(reg_c0, 8, v_acc01, false);
    store_acc(reg_c0, 16, v_acc02, false);
    store_acc(reg_c0, 24, v_acc03, false);
    store_acc(reg_c0, 32, v_acc04, false);
    store_acc(reg_c0, 40, v_acc05, false);
    store_acc(reg_c0, 48, v_acc06, false);
    store_acc(reg_c0, 56, v_acc07, false);

    store_acc(reg_c1, 0, v_acc00, true);
    store_acc(reg_c1, 8, v_acc01, true);
    store_acc(reg_c1, 16, v_acc02, true);
    store_acc(reg_c1, 24, v_acc03, true);
    store_acc(reg_c1, 32, v_acc04, true);
    store_acc(reg_c1, 40, v_acc05, true);
    store_acc(reg_c1, 48, v_acc06, true);
    store_acc(reg_c1, 56, v_acc07, true);

    store_acc(reg_c2, 0, v_acc10, false);
    store_acc(reg_c2, 8, v_acc11, false);
    store_acc(reg_c2, 16, v_acc12, false);
    store_acc(reg_c2, 24, v_acc13, false);
    store_acc(reg_c2, 32, v_acc14, false);
    store_acc(reg_c2, 40, v_acc15, false);
    store_acc(reg_c2, 48, v_acc16, false);
    store_acc(reg_c2, 56, v_acc17, false);

    store_acc(reg_c3, 0, v_acc10, true);
    store_acc(reg_c3, 8, v_acc11, true);
    store_acc(reg_c3, 16, v_acc12, true);
    store_acc(reg_c3, 24, v_acc13, true);
    store_acc(reg_c3, 32, v_acc14, true);
    store_acc(reg_c3, 40, v_acc15, true);
    store_acc(reg_c3, 48, v_acc16, true);
    store_acc(reg_c3, 56, v_acc17, true);
    b(store_done);

    L(store_no_accum);
    store_noacc(reg_c0, 0, v_acc00, false);
    store_noacc(reg_c0, 8, v_acc01, false);
    store_noacc(reg_c0, 16, v_acc02, false);
    store_noacc(reg_c0, 24, v_acc03, false);
    store_noacc(reg_c0, 32, v_acc04, false);
    store_noacc(reg_c0, 40, v_acc05, false);
    store_noacc(reg_c0, 48, v_acc06, false);
    store_noacc(reg_c0, 56, v_acc07, false);

    store_noacc(reg_c1, 0, v_acc00, true);
    store_noacc(reg_c1, 8, v_acc01, true);
    store_noacc(reg_c1, 16, v_acc02, true);
    store_noacc(reg_c1, 24, v_acc03, true);
    store_noacc(reg_c1, 32, v_acc04, true);
    store_noacc(reg_c1, 40, v_acc05, true);
    store_noacc(reg_c1, 48, v_acc06, true);
    store_noacc(reg_c1, 56, v_acc07, true);

    store_noacc(reg_c2, 0, v_acc10, false);
    store_noacc(reg_c2, 8, v_acc11, false);
    store_noacc(reg_c2, 16, v_acc12, false);
    store_noacc(reg_c2, 24, v_acc13, false);
    store_noacc(reg_c2, 32, v_acc14, false);
    store_noacc(reg_c2, 40, v_acc15, false);
    store_noacc(reg_c2, 48, v_acc16, false);
    store_noacc(reg_c2, 56, v_acc17, false);

    store_noacc(reg_c3, 0, v_acc10, true);
    store_noacc(reg_c3, 8, v_acc11, true);
    store_noacc(reg_c3, 16, v_acc12, true);
    store_noacc(reg_c3, 24, v_acc13, true);
    store_noacc(reg_c3, 32, v_acc14, true);
    store_noacc(reg_c3, 40, v_acc15, true);
    store_noacc(reg_c3, 48, v_acc16, true);
    store_noacc(reg_c3, 56, v_acc17, true);

    L(store_done);

    L(done);
    postamble();
}

jit_int8_brgemm_kernel_4x4_usmmla_packed::jit_int8_brgemm_kernel_4x4_usmmla_packed() : jit_generator() {}

void jit_int8_brgemm_kernel_4x4_usmmla_packed::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_4x4_usmmla_packed::generate() {
    preamble();

    const XReg reg_srcs = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_ldc = abi_param6;
    const XReg reg_accum = abi_param7;

    const XReg reg_src0 = x10;
    const XReg reg_src1 = x11;
    const XReg reg_src2 = x12;
    const XReg reg_src3 = x13;

    const XReg reg_c0 = x18;
    const XReg reg_c1 = x19;
    const XReg reg_c2 = x20;
    const XReg reg_c3 = x21;

    const VReg v_a01 = VReg(0);
    const VReg v_a23 = VReg(1);
    const VReg v_tmp0 = VReg(2);
    const VReg v_tmp1 = VReg(3);
    const VReg v_w01 = VReg(4);
    const VReg v_w23 = VReg(5);

    const VReg v_acc00 = VReg(16);
    const VReg v_acc01 = VReg(17);
    const VReg v_acc10 = VReg(18);
    const VReg v_acc11 = VReg(19);

    eor(v_acc00.b, v_acc00.b, v_acc00.b);
    eor(v_acc01.b, v_acc01.b, v_acc01.b);
    eor(v_acc10.b, v_acc10.b, v_acc10.b);
    eor(v_acc11.b, v_acc11.b, v_acc11.b);

    ldr(reg_src0, ptr(reg_srcs));
    ldr(reg_src1, ptr(reg_srcs, 8));
    ldr(reg_src2, ptr(reg_srcs, 16));
    ldr(reg_src3, ptr(reg_srcs, 24));

    mov(reg_c0, reg_dst);
    add(reg_c1, reg_c0, reg_ldc);
    add(reg_c2, reg_c1, reg_ldc);
    add(reg_c3, reg_c2, reg_ldc);

    auto emit_usmmla = [&](const VReg4S& vd, const VReg16B& vn, const VReg16B& vm) {
        const uint32_t base = 0x4e80ac00;
        const uint32_t code = base | static_cast<uint32_t>(vd.getIdx()) |
                              (static_cast<uint32_t>(vn.getIdx()) << 5) |
                              (static_cast<uint32_t>(vm.getIdx()) << 16);
        dd(code);
    };

    Label loop16;
    Label loop8;
    Label done;

    cbz(reg_k, done);

    cmp(reg_k, 16);
    b(LT, loop8);

    L(loop16);
    ldr(DReg(v_a01.getIdx()), ptr(reg_src0));
    ldr(DReg(v_tmp0.getIdx()), ptr(reg_src1));
    ins(v_a01.d[1], v_tmp0.d[0]);
    ldr(DReg(v_a23.getIdx()), ptr(reg_src2));
    ldr(DReg(v_tmp1.getIdx()), ptr(reg_src3));
    ins(v_a23.d[1], v_tmp1.d[0]);

    ldr(QReg(v_w01.getIdx()), ptr(reg_wei));
    ldr(QReg(v_w23.getIdx()), ptr(reg_wei, 16));

    emit_usmmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_usmmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_usmmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_usmmla(v_acc11.s, v_a23.b, v_w23.b);

    add(reg_src0, reg_src0, 8);
    add(reg_src1, reg_src1, 8);
    add(reg_src2, reg_src2, 8);
    add(reg_src3, reg_src3, 8);
    add(reg_wei, reg_wei, 32);

    ldr(DReg(v_a01.getIdx()), ptr(reg_src0));
    ldr(DReg(v_tmp0.getIdx()), ptr(reg_src1));
    ins(v_a01.d[1], v_tmp0.d[0]);
    ldr(DReg(v_a23.getIdx()), ptr(reg_src2));
    ldr(DReg(v_tmp1.getIdx()), ptr(reg_src3));
    ins(v_a23.d[1], v_tmp1.d[0]);

    ldr(QReg(v_w01.getIdx()), ptr(reg_wei));
    ldr(QReg(v_w23.getIdx()), ptr(reg_wei, 16));

    emit_usmmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_usmmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_usmmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_usmmla(v_acc11.s, v_a23.b, v_w23.b);

    add(reg_src0, reg_src0, 8);
    add(reg_src1, reg_src1, 8);
    add(reg_src2, reg_src2, 8);
    add(reg_src3, reg_src3, 8);
    add(reg_wei, reg_wei, 32);

    subs(reg_k, reg_k, 16);
    cmp(reg_k, 16);
    b(GE, loop16);

    L(loop8);
    cbz(reg_k, done);
    ldr(DReg(v_a01.getIdx()), ptr(reg_src0));
    ldr(DReg(v_tmp0.getIdx()), ptr(reg_src1));
    ins(v_a01.d[1], v_tmp0.d[0]);
    ldr(DReg(v_a23.getIdx()), ptr(reg_src2));
    ldr(DReg(v_tmp1.getIdx()), ptr(reg_src3));
    ins(v_a23.d[1], v_tmp1.d[0]);

    ldr(QReg(v_w01.getIdx()), ptr(reg_wei));
    ldr(QReg(v_w23.getIdx()), ptr(reg_wei, 16));

    emit_usmmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_usmmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_usmmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_usmmla(v_acc11.s, v_a23.b, v_w23.b);

    add(reg_src0, reg_src0, 8);
    add(reg_src1, reg_src1, 8);
    add(reg_src2, reg_src2, 8);
    add(reg_src3, reg_src3, 8);
    add(reg_wei, reg_wei, 32);

    subs(reg_k, reg_k, 8);
    // reg_k is expected to hit zero here

    const WReg reg_tmp = w9;
    const WReg reg_c_acc = w0;
    Label store_no_accum;
    Label store_accum;
    Label store_done;

    cbz(reg_accum, store_no_accum);

    L(store_accum);
    umov(reg_tmp, v_acc00.s[0]);
    ldr(reg_c_acc, ptr(reg_c0));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0));
    umov(reg_tmp, v_acc00.s[1]);
    ldr(reg_c_acc, ptr(reg_c0, 4));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0, 4));
    umov(reg_tmp, v_acc01.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 8));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0, 8));
    umov(reg_tmp, v_acc01.s[1]);
    ldr(reg_c_acc, ptr(reg_c0, 12));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0, 12));

    umov(reg_tmp, v_acc00.s[2]);
    ldr(reg_c_acc, ptr(reg_c1));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1));
    umov(reg_tmp, v_acc00.s[3]);
    ldr(reg_c_acc, ptr(reg_c1, 4));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1, 4));
    umov(reg_tmp, v_acc01.s[2]);
    ldr(reg_c_acc, ptr(reg_c1, 8));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1, 8));
    umov(reg_tmp, v_acc01.s[3]);
    ldr(reg_c_acc, ptr(reg_c1, 12));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1, 12));

    umov(reg_tmp, v_acc10.s[0]);
    ldr(reg_c_acc, ptr(reg_c2));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2));
    umov(reg_tmp, v_acc10.s[1]);
    ldr(reg_c_acc, ptr(reg_c2, 4));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2, 4));
    umov(reg_tmp, v_acc11.s[0]);
    ldr(reg_c_acc, ptr(reg_c2, 8));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2, 8));
    umov(reg_tmp, v_acc11.s[1]);
    ldr(reg_c_acc, ptr(reg_c2, 12));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2, 12));

    umov(reg_tmp, v_acc10.s[2]);
    ldr(reg_c_acc, ptr(reg_c3));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3));
    umov(reg_tmp, v_acc10.s[3]);
    ldr(reg_c_acc, ptr(reg_c3, 4));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3, 4));
    umov(reg_tmp, v_acc11.s[2]);
    ldr(reg_c_acc, ptr(reg_c3, 8));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3, 8));
    umov(reg_tmp, v_acc11.s[3]);
    ldr(reg_c_acc, ptr(reg_c3, 12));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3, 12));
    b(store_done);

    L(store_no_accum);
    umov(reg_tmp, v_acc00.s[0]);
    str(reg_tmp, ptr(reg_c0));
    umov(reg_tmp, v_acc00.s[1]);
    str(reg_tmp, ptr(reg_c0, 4));
    umov(reg_tmp, v_acc01.s[0]);
    str(reg_tmp, ptr(reg_c0, 8));
    umov(reg_tmp, v_acc01.s[1]);
    str(reg_tmp, ptr(reg_c0, 12));

    umov(reg_tmp, v_acc00.s[2]);
    str(reg_tmp, ptr(reg_c1));
    umov(reg_tmp, v_acc00.s[3]);
    str(reg_tmp, ptr(reg_c1, 4));
    umov(reg_tmp, v_acc01.s[2]);
    str(reg_tmp, ptr(reg_c1, 8));
    umov(reg_tmp, v_acc01.s[3]);
    str(reg_tmp, ptr(reg_c1, 12));

    umov(reg_tmp, v_acc10.s[0]);
    str(reg_tmp, ptr(reg_c2));
    umov(reg_tmp, v_acc10.s[1]);
    str(reg_tmp, ptr(reg_c2, 4));
    umov(reg_tmp, v_acc11.s[0]);
    str(reg_tmp, ptr(reg_c2, 8));
    umov(reg_tmp, v_acc11.s[1]);
    str(reg_tmp, ptr(reg_c2, 12));

    umov(reg_tmp, v_acc10.s[2]);
    str(reg_tmp, ptr(reg_c3));
    umov(reg_tmp, v_acc10.s[3]);
    str(reg_tmp, ptr(reg_c3, 4));
    umov(reg_tmp, v_acc11.s[2]);
    str(reg_tmp, ptr(reg_c3, 8));
    umov(reg_tmp, v_acc11.s[3]);
    str(reg_tmp, ptr(reg_c3, 12));

    L(store_done);

    L(done);
    postamble();
}

jit_int8_brgemm_kernel_4x8_usmmla_packed::jit_int8_brgemm_kernel_4x8_usmmla_packed() : jit_generator() {}

void jit_int8_brgemm_kernel_4x8_usmmla_packed::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_4x8_usmmla_packed::generate() {
    preamble();

    const XReg reg_srcs = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_ldc = abi_param6;
    const XReg reg_accum = abi_param7;

    const XReg reg_src0 = x10;
    const XReg reg_src1 = x11;
    const XReg reg_src2 = x12;
    const XReg reg_src3 = x13;

    const XReg reg_c0 = x18;
    const XReg reg_c1 = x19;
    const XReg reg_c2 = x20;
    const XReg reg_c3 = x21;

    const VReg v_a01 = VReg(0);
    const VReg v_a23 = VReg(1);
    const VReg v_tmp0 = VReg(2);
    const VReg v_tmp1 = VReg(3);
    const VReg v_w01 = VReg(4);
    const VReg v_w23 = VReg(5);
    const VReg v_w45 = VReg(6);
    const VReg v_w67 = VReg(7);

    const VReg v_acc00 = VReg(16);
    const VReg v_acc01 = VReg(17);
    const VReg v_acc02 = VReg(18);
    const VReg v_acc03 = VReg(19);
    const VReg v_acc10 = VReg(20);
    const VReg v_acc11 = VReg(21);
    const VReg v_acc12 = VReg(22);
    const VReg v_acc13 = VReg(23);

    eor(v_acc00.b, v_acc00.b, v_acc00.b);
    eor(v_acc01.b, v_acc01.b, v_acc01.b);
    eor(v_acc02.b, v_acc02.b, v_acc02.b);
    eor(v_acc03.b, v_acc03.b, v_acc03.b);
    eor(v_acc10.b, v_acc10.b, v_acc10.b);
    eor(v_acc11.b, v_acc11.b, v_acc11.b);
    eor(v_acc12.b, v_acc12.b, v_acc12.b);
    eor(v_acc13.b, v_acc13.b, v_acc13.b);

    ldr(reg_src0, ptr(reg_srcs));
    ldr(reg_src1, ptr(reg_srcs, 8));
    ldr(reg_src2, ptr(reg_srcs, 16));
    ldr(reg_src3, ptr(reg_srcs, 24));

    mov(reg_c0, reg_dst);
    add(reg_c1, reg_c0, reg_ldc);
    add(reg_c2, reg_c1, reg_ldc);
    add(reg_c3, reg_c2, reg_ldc);

    auto emit_usmmla = [&](const VReg4S& vd, const VReg16B& vn, const VReg16B& vm) {
        const uint32_t base = 0x4e80ac00;
        const uint32_t code = base | static_cast<uint32_t>(vd.getIdx()) |
                              (static_cast<uint32_t>(vn.getIdx()) << 5) |
                              (static_cast<uint32_t>(vm.getIdx()) << 16);
        dd(code);
    };

    Label loop16;
    Label loop8;
    Label done;

    cbz(reg_k, done);

    cmp(reg_k, 16);
    b(LT, loop8);

    L(loop16);
    ldr(DReg(v_a01.getIdx()), ptr(reg_src0));
    ldr(DReg(v_tmp0.getIdx()), ptr(reg_src1));
    ins(v_a01.d[1], v_tmp0.d[0]);
    ldr(DReg(v_a23.getIdx()), ptr(reg_src2));
    ldr(DReg(v_tmp1.getIdx()), ptr(reg_src3));
    ins(v_a23.d[1], v_tmp1.d[0]);

    ldr(QReg(v_w01.getIdx()), ptr(reg_wei));
    ldr(QReg(v_w23.getIdx()), ptr(reg_wei, 16));
    ldr(QReg(v_w45.getIdx()), ptr(reg_wei, 32));
    ldr(QReg(v_w67.getIdx()), ptr(reg_wei, 48));

    emit_usmmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_usmmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_usmmla(v_acc02.s, v_a01.b, v_w45.b);
    emit_usmmla(v_acc03.s, v_a01.b, v_w67.b);
    emit_usmmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_usmmla(v_acc11.s, v_a23.b, v_w23.b);
    emit_usmmla(v_acc12.s, v_a23.b, v_w45.b);
    emit_usmmla(v_acc13.s, v_a23.b, v_w67.b);

    add(reg_src0, reg_src0, 8);
    add(reg_src1, reg_src1, 8);
    add(reg_src2, reg_src2, 8);
    add(reg_src3, reg_src3, 8);
    add(reg_wei, reg_wei, 64);

    ldr(DReg(v_a01.getIdx()), ptr(reg_src0));
    ldr(DReg(v_tmp0.getIdx()), ptr(reg_src1));
    ins(v_a01.d[1], v_tmp0.d[0]);
    ldr(DReg(v_a23.getIdx()), ptr(reg_src2));
    ldr(DReg(v_tmp1.getIdx()), ptr(reg_src3));
    ins(v_a23.d[1], v_tmp1.d[0]);

    ldr(QReg(v_w01.getIdx()), ptr(reg_wei));
    ldr(QReg(v_w23.getIdx()), ptr(reg_wei, 16));
    ldr(QReg(v_w45.getIdx()), ptr(reg_wei, 32));
    ldr(QReg(v_w67.getIdx()), ptr(reg_wei, 48));

    emit_usmmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_usmmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_usmmla(v_acc02.s, v_a01.b, v_w45.b);
    emit_usmmla(v_acc03.s, v_a01.b, v_w67.b);
    emit_usmmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_usmmla(v_acc11.s, v_a23.b, v_w23.b);
    emit_usmmla(v_acc12.s, v_a23.b, v_w45.b);
    emit_usmmla(v_acc13.s, v_a23.b, v_w67.b);

    add(reg_src0, reg_src0, 8);
    add(reg_src1, reg_src1, 8);
    add(reg_src2, reg_src2, 8);
    add(reg_src3, reg_src3, 8);
    add(reg_wei, reg_wei, 64);

    subs(reg_k, reg_k, 16);
    cmp(reg_k, 16);
    b(GE, loop16);

    L(loop8);
    cbz(reg_k, done);
    ldr(DReg(v_a01.getIdx()), ptr(reg_src0));
    ldr(DReg(v_tmp0.getIdx()), ptr(reg_src1));
    ins(v_a01.d[1], v_tmp0.d[0]);
    ldr(DReg(v_a23.getIdx()), ptr(reg_src2));
    ldr(DReg(v_tmp1.getIdx()), ptr(reg_src3));
    ins(v_a23.d[1], v_tmp1.d[0]);

    ldr(QReg(v_w01.getIdx()), ptr(reg_wei));
    ldr(QReg(v_w23.getIdx()), ptr(reg_wei, 16));
    ldr(QReg(v_w45.getIdx()), ptr(reg_wei, 32));
    ldr(QReg(v_w67.getIdx()), ptr(reg_wei, 48));

    emit_usmmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_usmmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_usmmla(v_acc02.s, v_a01.b, v_w45.b);
    emit_usmmla(v_acc03.s, v_a01.b, v_w67.b);
    emit_usmmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_usmmla(v_acc11.s, v_a23.b, v_w23.b);
    emit_usmmla(v_acc12.s, v_a23.b, v_w45.b);
    emit_usmmla(v_acc13.s, v_a23.b, v_w67.b);

    add(reg_src0, reg_src0, 8);
    add(reg_src1, reg_src1, 8);
    add(reg_src2, reg_src2, 8);
    add(reg_src3, reg_src3, 8);
    add(reg_wei, reg_wei, 64);

    subs(reg_k, reg_k, 8);
    // reg_k is expected to hit zero here

    const WReg reg_tmp = w9;
    const WReg reg_c_acc = w0;
    Label store_no_accum;
    Label store_accum;
    Label store_done;

    cbz(reg_accum, store_no_accum);

    L(store_accum);
    umov(reg_tmp, v_acc00.s[0]);
    ldr(reg_c_acc, ptr(reg_c0));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0));
    umov(reg_tmp, v_acc00.s[1]);
    ldr(reg_c_acc, ptr(reg_c0, 4));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0, 4));
    umov(reg_tmp, v_acc01.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 8));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0, 8));
    umov(reg_tmp, v_acc01.s[1]);
    ldr(reg_c_acc, ptr(reg_c0, 12));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0, 12));
    umov(reg_tmp, v_acc02.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 16));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0, 16));
    umov(reg_tmp, v_acc02.s[1]);
    ldr(reg_c_acc, ptr(reg_c0, 20));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0, 20));
    umov(reg_tmp, v_acc03.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 24));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0, 24));
    umov(reg_tmp, v_acc03.s[1]);
    ldr(reg_c_acc, ptr(reg_c0, 28));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c0, 28));

    umov(reg_tmp, v_acc00.s[2]);
    ldr(reg_c_acc, ptr(reg_c1));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1));
    umov(reg_tmp, v_acc00.s[3]);
    ldr(reg_c_acc, ptr(reg_c1, 4));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1, 4));
    umov(reg_tmp, v_acc01.s[2]);
    ldr(reg_c_acc, ptr(reg_c1, 8));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1, 8));
    umov(reg_tmp, v_acc01.s[3]);
    ldr(reg_c_acc, ptr(reg_c1, 12));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1, 12));
    umov(reg_tmp, v_acc02.s[2]);
    ldr(reg_c_acc, ptr(reg_c1, 16));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1, 16));
    umov(reg_tmp, v_acc02.s[3]);
    ldr(reg_c_acc, ptr(reg_c1, 20));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1, 20));
    umov(reg_tmp, v_acc03.s[2]);
    ldr(reg_c_acc, ptr(reg_c1, 24));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1, 24));
    umov(reg_tmp, v_acc03.s[3]);
    ldr(reg_c_acc, ptr(reg_c1, 28));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c1, 28));

    umov(reg_tmp, v_acc10.s[0]);
    ldr(reg_c_acc, ptr(reg_c2));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2));
    umov(reg_tmp, v_acc10.s[1]);
    ldr(reg_c_acc, ptr(reg_c2, 4));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2, 4));
    umov(reg_tmp, v_acc11.s[0]);
    ldr(reg_c_acc, ptr(reg_c2, 8));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2, 8));
    umov(reg_tmp, v_acc11.s[1]);
    ldr(reg_c_acc, ptr(reg_c2, 12));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2, 12));
    umov(reg_tmp, v_acc12.s[0]);
    ldr(reg_c_acc, ptr(reg_c2, 16));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2, 16));
    umov(reg_tmp, v_acc12.s[1]);
    ldr(reg_c_acc, ptr(reg_c2, 20));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2, 20));
    umov(reg_tmp, v_acc13.s[0]);
    ldr(reg_c_acc, ptr(reg_c2, 24));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2, 24));
    umov(reg_tmp, v_acc13.s[1]);
    ldr(reg_c_acc, ptr(reg_c2, 28));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c2, 28));

    umov(reg_tmp, v_acc10.s[2]);
    ldr(reg_c_acc, ptr(reg_c3));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3));
    umov(reg_tmp, v_acc10.s[3]);
    ldr(reg_c_acc, ptr(reg_c3, 4));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3, 4));
    umov(reg_tmp, v_acc11.s[2]);
    ldr(reg_c_acc, ptr(reg_c3, 8));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3, 8));
    umov(reg_tmp, v_acc11.s[3]);
    ldr(reg_c_acc, ptr(reg_c3, 12));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3, 12));
    umov(reg_tmp, v_acc12.s[2]);
    ldr(reg_c_acc, ptr(reg_c3, 16));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3, 16));
    umov(reg_tmp, v_acc12.s[3]);
    ldr(reg_c_acc, ptr(reg_c3, 20));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3, 20));
    umov(reg_tmp, v_acc13.s[2]);
    ldr(reg_c_acc, ptr(reg_c3, 24));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3, 24));
    umov(reg_tmp, v_acc13.s[3]);
    ldr(reg_c_acc, ptr(reg_c3, 28));
    add(reg_tmp, reg_tmp, reg_c_acc);
    str(reg_tmp, ptr(reg_c3, 28));
    b(store_done);

    L(store_no_accum);
    umov(reg_tmp, v_acc00.s[0]);
    str(reg_tmp, ptr(reg_c0));
    umov(reg_tmp, v_acc00.s[1]);
    str(reg_tmp, ptr(reg_c0, 4));
    umov(reg_tmp, v_acc01.s[0]);
    str(reg_tmp, ptr(reg_c0, 8));
    umov(reg_tmp, v_acc01.s[1]);
    str(reg_tmp, ptr(reg_c0, 12));
    umov(reg_tmp, v_acc02.s[0]);
    str(reg_tmp, ptr(reg_c0, 16));
    umov(reg_tmp, v_acc02.s[1]);
    str(reg_tmp, ptr(reg_c0, 20));
    umov(reg_tmp, v_acc03.s[0]);
    str(reg_tmp, ptr(reg_c0, 24));
    umov(reg_tmp, v_acc03.s[1]);
    str(reg_tmp, ptr(reg_c0, 28));

    umov(reg_tmp, v_acc00.s[2]);
    str(reg_tmp, ptr(reg_c1));
    umov(reg_tmp, v_acc00.s[3]);
    str(reg_tmp, ptr(reg_c1, 4));
    umov(reg_tmp, v_acc01.s[2]);
    str(reg_tmp, ptr(reg_c1, 8));
    umov(reg_tmp, v_acc01.s[3]);
    str(reg_tmp, ptr(reg_c1, 12));
    umov(reg_tmp, v_acc02.s[2]);
    str(reg_tmp, ptr(reg_c1, 16));
    umov(reg_tmp, v_acc02.s[3]);
    str(reg_tmp, ptr(reg_c1, 20));
    umov(reg_tmp, v_acc03.s[2]);
    str(reg_tmp, ptr(reg_c1, 24));
    umov(reg_tmp, v_acc03.s[3]);
    str(reg_tmp, ptr(reg_c1, 28));

    umov(reg_tmp, v_acc10.s[0]);
    str(reg_tmp, ptr(reg_c2));
    umov(reg_tmp, v_acc10.s[1]);
    str(reg_tmp, ptr(reg_c2, 4));
    umov(reg_tmp, v_acc11.s[0]);
    str(reg_tmp, ptr(reg_c2, 8));
    umov(reg_tmp, v_acc11.s[1]);
    str(reg_tmp, ptr(reg_c2, 12));
    umov(reg_tmp, v_acc12.s[0]);
    str(reg_tmp, ptr(reg_c2, 16));
    umov(reg_tmp, v_acc12.s[1]);
    str(reg_tmp, ptr(reg_c2, 20));
    umov(reg_tmp, v_acc13.s[0]);
    str(reg_tmp, ptr(reg_c2, 24));
    umov(reg_tmp, v_acc13.s[1]);
    str(reg_tmp, ptr(reg_c2, 28));

    umov(reg_tmp, v_acc10.s[2]);
    str(reg_tmp, ptr(reg_c3));
    umov(reg_tmp, v_acc10.s[3]);
    str(reg_tmp, ptr(reg_c3, 4));
    umov(reg_tmp, v_acc11.s[2]);
    str(reg_tmp, ptr(reg_c3, 8));
    umov(reg_tmp, v_acc11.s[3]);
    str(reg_tmp, ptr(reg_c3, 12));
    umov(reg_tmp, v_acc12.s[2]);
    str(reg_tmp, ptr(reg_c3, 16));
    umov(reg_tmp, v_acc12.s[3]);
    str(reg_tmp, ptr(reg_c3, 20));
    umov(reg_tmp, v_acc13.s[2]);
    str(reg_tmp, ptr(reg_c3, 24));
    umov(reg_tmp, v_acc13.s[3]);
    str(reg_tmp, ptr(reg_c3, 28));

    L(store_done);

    L(done);
    postamble();
}

jit_int8_brgemm_kernel_4x16_usmmla_packed::jit_int8_brgemm_kernel_4x16_usmmla_packed() : jit_generator() {}

void jit_int8_brgemm_kernel_4x16_usmmla_packed::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_4x16_usmmla_packed::generate() {
    preamble();

    const XReg reg_srcs = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_ldc = abi_param6;
    const XReg reg_accum = abi_param7;

    const XReg reg_src0 = x10;
    const XReg reg_src1 = x11;
    const XReg reg_src2 = x12;
    const XReg reg_src3 = x13;

    const XReg reg_c0 = x18;
    const XReg reg_c1 = x19;
    const XReg reg_c2 = x20;
    const XReg reg_c3 = x21;

    const VReg v_a01 = VReg(0);
    const VReg v_a23 = VReg(1);
    const VReg v_tmp0 = VReg(2);
    const VReg v_tmp1 = VReg(3);
    const VReg v_w01 = VReg(4);
    const VReg v_w23 = VReg(5);
    const VReg v_w45 = VReg(6);
    const VReg v_w67 = VReg(7);
    const VReg v_w89 = VReg(8);
    const VReg v_wab = VReg(9);
    const VReg v_wcd = VReg(10);
    const VReg v_wef = VReg(11);

    const VReg v_acc00 = VReg(16);
    const VReg v_acc01 = VReg(17);
    const VReg v_acc02 = VReg(18);
    const VReg v_acc03 = VReg(19);
    const VReg v_acc04 = VReg(20);
    const VReg v_acc05 = VReg(21);
    const VReg v_acc06 = VReg(22);
    const VReg v_acc07 = VReg(23);
    const VReg v_acc10 = VReg(24);
    const VReg v_acc11 = VReg(25);
    const VReg v_acc12 = VReg(26);
    const VReg v_acc13 = VReg(27);
    const VReg v_acc14 = VReg(28);
    const VReg v_acc15 = VReg(29);
    const VReg v_acc16 = VReg(30);
    const VReg v_acc17 = VReg(31);

    eor(v_acc00.b, v_acc00.b, v_acc00.b);
    eor(v_acc01.b, v_acc01.b, v_acc01.b);
    eor(v_acc02.b, v_acc02.b, v_acc02.b);
    eor(v_acc03.b, v_acc03.b, v_acc03.b);
    eor(v_acc04.b, v_acc04.b, v_acc04.b);
    eor(v_acc05.b, v_acc05.b, v_acc05.b);
    eor(v_acc06.b, v_acc06.b, v_acc06.b);
    eor(v_acc07.b, v_acc07.b, v_acc07.b);
    eor(v_acc10.b, v_acc10.b, v_acc10.b);
    eor(v_acc11.b, v_acc11.b, v_acc11.b);
    eor(v_acc12.b, v_acc12.b, v_acc12.b);
    eor(v_acc13.b, v_acc13.b, v_acc13.b);
    eor(v_acc14.b, v_acc14.b, v_acc14.b);
    eor(v_acc15.b, v_acc15.b, v_acc15.b);
    eor(v_acc16.b, v_acc16.b, v_acc16.b);
    eor(v_acc17.b, v_acc17.b, v_acc17.b);

    ldr(reg_src0, ptr(reg_srcs));
    ldr(reg_src1, ptr(reg_srcs, 8));
    ldr(reg_src2, ptr(reg_srcs, 16));
    ldr(reg_src3, ptr(reg_srcs, 24));

    mov(reg_c0, reg_dst);
    add(reg_c1, reg_c0, reg_ldc);
    add(reg_c2, reg_c1, reg_ldc);
    add(reg_c3, reg_c2, reg_ldc);

    auto emit_usmmla = [&](const VReg4S& vd, const VReg16B& vn, const VReg16B& vm) {
        const uint32_t base = 0x4e80ac00;
        const uint32_t code = base | static_cast<uint32_t>(vd.getIdx()) |
                              (static_cast<uint32_t>(vn.getIdx()) << 5) |
                              (static_cast<uint32_t>(vm.getIdx()) << 16);
        dd(code);
    };

    Label loop16;
    Label loop8;
    Label done;

    cbz(reg_k, done);

    cmp(reg_k, 16);
    b(LT, loop8);

    L(loop16);
    ldr(DReg(v_a01.getIdx()), ptr(reg_src0));
    ldr(DReg(v_tmp0.getIdx()), ptr(reg_src1));
    ins(v_a01.d[1], v_tmp0.d[0]);
    ldr(DReg(v_a23.getIdx()), ptr(reg_src2));
    ldr(DReg(v_tmp1.getIdx()), ptr(reg_src3));
    ins(v_a23.d[1], v_tmp1.d[0]);

    ldr(QReg(v_w01.getIdx()), ptr(reg_wei));
    ldr(QReg(v_w23.getIdx()), ptr(reg_wei, 16));
    ldr(QReg(v_w45.getIdx()), ptr(reg_wei, 32));
    ldr(QReg(v_w67.getIdx()), ptr(reg_wei, 48));
    ldr(QReg(v_w89.getIdx()), ptr(reg_wei, 64));
    ldr(QReg(v_wab.getIdx()), ptr(reg_wei, 80));
    ldr(QReg(v_wcd.getIdx()), ptr(reg_wei, 96));
    ldr(QReg(v_wef.getIdx()), ptr(reg_wei, 112));

    emit_usmmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_usmmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_usmmla(v_acc02.s, v_a01.b, v_w45.b);
    emit_usmmla(v_acc03.s, v_a01.b, v_w67.b);
    emit_usmmla(v_acc04.s, v_a01.b, v_w89.b);
    emit_usmmla(v_acc05.s, v_a01.b, v_wab.b);
    emit_usmmla(v_acc06.s, v_a01.b, v_wcd.b);
    emit_usmmla(v_acc07.s, v_a01.b, v_wef.b);
    emit_usmmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_usmmla(v_acc11.s, v_a23.b, v_w23.b);
    emit_usmmla(v_acc12.s, v_a23.b, v_w45.b);
    emit_usmmla(v_acc13.s, v_a23.b, v_w67.b);
    emit_usmmla(v_acc14.s, v_a23.b, v_w89.b);
    emit_usmmla(v_acc15.s, v_a23.b, v_wab.b);
    emit_usmmla(v_acc16.s, v_a23.b, v_wcd.b);
    emit_usmmla(v_acc17.s, v_a23.b, v_wef.b);

    add(reg_src0, reg_src0, 8);
    add(reg_src1, reg_src1, 8);
    add(reg_src2, reg_src2, 8);
    add(reg_src3, reg_src3, 8);
    add(reg_wei, reg_wei, 128);

    ldr(DReg(v_a01.getIdx()), ptr(reg_src0));
    ldr(DReg(v_tmp0.getIdx()), ptr(reg_src1));
    ins(v_a01.d[1], v_tmp0.d[0]);
    ldr(DReg(v_a23.getIdx()), ptr(reg_src2));
    ldr(DReg(v_tmp1.getIdx()), ptr(reg_src3));
    ins(v_a23.d[1], v_tmp1.d[0]);

    ldr(QReg(v_w01.getIdx()), ptr(reg_wei));
    ldr(QReg(v_w23.getIdx()), ptr(reg_wei, 16));
    ldr(QReg(v_w45.getIdx()), ptr(reg_wei, 32));
    ldr(QReg(v_w67.getIdx()), ptr(reg_wei, 48));
    ldr(QReg(v_w89.getIdx()), ptr(reg_wei, 64));
    ldr(QReg(v_wab.getIdx()), ptr(reg_wei, 80));
    ldr(QReg(v_wcd.getIdx()), ptr(reg_wei, 96));
    ldr(QReg(v_wef.getIdx()), ptr(reg_wei, 112));

    emit_usmmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_usmmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_usmmla(v_acc02.s, v_a01.b, v_w45.b);
    emit_usmmla(v_acc03.s, v_a01.b, v_w67.b);
    emit_usmmla(v_acc04.s, v_a01.b, v_w89.b);
    emit_usmmla(v_acc05.s, v_a01.b, v_wab.b);
    emit_usmmla(v_acc06.s, v_a01.b, v_wcd.b);
    emit_usmmla(v_acc07.s, v_a01.b, v_wef.b);
    emit_usmmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_usmmla(v_acc11.s, v_a23.b, v_w23.b);
    emit_usmmla(v_acc12.s, v_a23.b, v_w45.b);
    emit_usmmla(v_acc13.s, v_a23.b, v_w67.b);
    emit_usmmla(v_acc14.s, v_a23.b, v_w89.b);
    emit_usmmla(v_acc15.s, v_a23.b, v_wab.b);
    emit_usmmla(v_acc16.s, v_a23.b, v_wcd.b);
    emit_usmmla(v_acc17.s, v_a23.b, v_wef.b);

    add(reg_src0, reg_src0, 8);
    add(reg_src1, reg_src1, 8);
    add(reg_src2, reg_src2, 8);
    add(reg_src3, reg_src3, 8);
    add(reg_wei, reg_wei, 128);

    subs(reg_k, reg_k, 16);
    cmp(reg_k, 16);
    b(GE, loop16);

    L(loop8);
    cbz(reg_k, done);
    ldr(DReg(v_a01.getIdx()), ptr(reg_src0));
    ldr(DReg(v_tmp0.getIdx()), ptr(reg_src1));
    ins(v_a01.d[1], v_tmp0.d[0]);
    ldr(DReg(v_a23.getIdx()), ptr(reg_src2));
    ldr(DReg(v_tmp1.getIdx()), ptr(reg_src3));
    ins(v_a23.d[1], v_tmp1.d[0]);

    ldr(QReg(v_w01.getIdx()), ptr(reg_wei));
    ldr(QReg(v_w23.getIdx()), ptr(reg_wei, 16));
    ldr(QReg(v_w45.getIdx()), ptr(reg_wei, 32));
    ldr(QReg(v_w67.getIdx()), ptr(reg_wei, 48));
    ldr(QReg(v_w89.getIdx()), ptr(reg_wei, 64));
    ldr(QReg(v_wab.getIdx()), ptr(reg_wei, 80));
    ldr(QReg(v_wcd.getIdx()), ptr(reg_wei, 96));
    ldr(QReg(v_wef.getIdx()), ptr(reg_wei, 112));

    emit_usmmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_usmmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_usmmla(v_acc02.s, v_a01.b, v_w45.b);
    emit_usmmla(v_acc03.s, v_a01.b, v_w67.b);
    emit_usmmla(v_acc04.s, v_a01.b, v_w89.b);
    emit_usmmla(v_acc05.s, v_a01.b, v_wab.b);
    emit_usmmla(v_acc06.s, v_a01.b, v_wcd.b);
    emit_usmmla(v_acc07.s, v_a01.b, v_wef.b);
    emit_usmmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_usmmla(v_acc11.s, v_a23.b, v_w23.b);
    emit_usmmla(v_acc12.s, v_a23.b, v_w45.b);
    emit_usmmla(v_acc13.s, v_a23.b, v_w67.b);
    emit_usmmla(v_acc14.s, v_a23.b, v_w89.b);
    emit_usmmla(v_acc15.s, v_a23.b, v_wab.b);
    emit_usmmla(v_acc16.s, v_a23.b, v_wcd.b);
    emit_usmmla(v_acc17.s, v_a23.b, v_wef.b);

    add(reg_src0, reg_src0, 8);
    add(reg_src1, reg_src1, 8);
    add(reg_src2, reg_src2, 8);
    add(reg_src3, reg_src3, 8);
    add(reg_wei, reg_wei, 128);

    subs(reg_k, reg_k, 8);
    // reg_k is expected to hit zero here

    Label store_no_accum;
    Label store_accum;
    Label store_done;

    auto store_acc = [&](const XReg& reg_c, int offset, const VReg& v_acc, bool upper) {
        ldr(DReg(v_tmp0.getIdx()), ptr(reg_c, offset));
        if (upper) {
            mov(v_tmp1.d[0], v_acc.d[1]);
            add(v_tmp0.s, v_tmp0.s, v_tmp1.s);
        } else {
            add(v_tmp0.s, v_tmp0.s, v_acc.s);
        }
        str(DReg(v_tmp0.getIdx()), ptr(reg_c, offset));
    };

    auto store_noacc = [&](const XReg& reg_c, int offset, const VReg& v_acc, bool upper) {
        if (upper) {
            mov(v_tmp1.d[0], v_acc.d[1]);
            str(DReg(v_tmp1.getIdx()), ptr(reg_c, offset));
        } else {
            str(DReg(v_acc.getIdx()), ptr(reg_c, offset));
        }
    };

    cbz(reg_accum, store_no_accum);

    L(store_accum);
    store_acc(reg_c0, 0, v_acc00, false);
    store_acc(reg_c0, 8, v_acc01, false);
    store_acc(reg_c0, 16, v_acc02, false);
    store_acc(reg_c0, 24, v_acc03, false);
    store_acc(reg_c0, 32, v_acc04, false);
    store_acc(reg_c0, 40, v_acc05, false);
    store_acc(reg_c0, 48, v_acc06, false);
    store_acc(reg_c0, 56, v_acc07, false);

    store_acc(reg_c1, 0, v_acc00, true);
    store_acc(reg_c1, 8, v_acc01, true);
    store_acc(reg_c1, 16, v_acc02, true);
    store_acc(reg_c1, 24, v_acc03, true);
    store_acc(reg_c1, 32, v_acc04, true);
    store_acc(reg_c1, 40, v_acc05, true);
    store_acc(reg_c1, 48, v_acc06, true);
    store_acc(reg_c1, 56, v_acc07, true);

    store_acc(reg_c2, 0, v_acc10, false);
    store_acc(reg_c2, 8, v_acc11, false);
    store_acc(reg_c2, 16, v_acc12, false);
    store_acc(reg_c2, 24, v_acc13, false);
    store_acc(reg_c2, 32, v_acc14, false);
    store_acc(reg_c2, 40, v_acc15, false);
    store_acc(reg_c2, 48, v_acc16, false);
    store_acc(reg_c2, 56, v_acc17, false);

    store_acc(reg_c3, 0, v_acc10, true);
    store_acc(reg_c3, 8, v_acc11, true);
    store_acc(reg_c3, 16, v_acc12, true);
    store_acc(reg_c3, 24, v_acc13, true);
    store_acc(reg_c3, 32, v_acc14, true);
    store_acc(reg_c3, 40, v_acc15, true);
    store_acc(reg_c3, 48, v_acc16, true);
    store_acc(reg_c3, 56, v_acc17, true);
    b(store_done);

    L(store_no_accum);
    store_noacc(reg_c0, 0, v_acc00, false);
    store_noacc(reg_c0, 8, v_acc01, false);
    store_noacc(reg_c0, 16, v_acc02, false);
    store_noacc(reg_c0, 24, v_acc03, false);
    store_noacc(reg_c0, 32, v_acc04, false);
    store_noacc(reg_c0, 40, v_acc05, false);
    store_noacc(reg_c0, 48, v_acc06, false);
    store_noacc(reg_c0, 56, v_acc07, false);

    store_noacc(reg_c1, 0, v_acc00, true);
    store_noacc(reg_c1, 8, v_acc01, true);
    store_noacc(reg_c1, 16, v_acc02, true);
    store_noacc(reg_c1, 24, v_acc03, true);
    store_noacc(reg_c1, 32, v_acc04, true);
    store_noacc(reg_c1, 40, v_acc05, true);
    store_noacc(reg_c1, 48, v_acc06, true);
    store_noacc(reg_c1, 56, v_acc07, true);

    store_noacc(reg_c2, 0, v_acc10, false);
    store_noacc(reg_c2, 8, v_acc11, false);
    store_noacc(reg_c2, 16, v_acc12, false);
    store_noacc(reg_c2, 24, v_acc13, false);
    store_noacc(reg_c2, 32, v_acc14, false);
    store_noacc(reg_c2, 40, v_acc15, false);
    store_noacc(reg_c2, 48, v_acc16, false);
    store_noacc(reg_c2, 56, v_acc17, false);

    store_noacc(reg_c3, 0, v_acc10, true);
    store_noacc(reg_c3, 8, v_acc11, true);
    store_noacc(reg_c3, 16, v_acc12, true);
    store_noacc(reg_c3, 24, v_acc13, true);
    store_noacc(reg_c3, 32, v_acc14, true);
    store_noacc(reg_c3, 40, v_acc15, true);
    store_noacc(reg_c3, 48, v_acc16, true);
    store_noacc(reg_c3, 56, v_acc17, true);

    L(store_done);

    L(done);
    postamble();
}

jit_int8_brgemm_kernel_4x4_udot_packed::jit_int8_brgemm_kernel_4x4_udot_packed() : jit_generator() {}

void jit_int8_brgemm_kernel_4x4_udot_packed::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_4x4_udot_packed::generate() {
    preamble();

    const XReg reg_srcs = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_ldc = abi_param6;
    const XReg reg_accum = abi_param7;

    const XReg reg_src0 = x10;
    const XReg reg_src1 = x11;
    const XReg reg_src2 = x12;
    const XReg reg_src3 = x13;

    const XReg reg_c0 = x18;
    const XReg reg_c1 = x19;
    const XReg reg_c2 = x20;
    const XReg reg_c3 = x21;

    const VReg16B v_src0 = VReg16B(0);
    const VReg16B v_src1 = VReg16B(1);
    const VReg16B v_src2 = VReg16B(2);
    const VReg16B v_src3 = VReg16B(3);
    const VReg16B v_w0 = VReg16B(4);
    const VReg16B v_w1 = VReg16B(5);
    const VReg16B v_w2 = VReg16B(6);
    const VReg16B v_w3 = VReg16B(7);
    const VReg16B v_mask = VReg16B(8);
    const VReg16BList v_wlist(v_w0, v_w3);

    const VReg4S v_acc00 = VReg4S(16);
    const VReg4S v_acc01 = VReg4S(17);
    const VReg4S v_acc02 = VReg4S(18);
    const VReg4S v_acc03 = VReg4S(19);
    const VReg4S v_acc10 = VReg4S(20);
    const VReg4S v_acc11 = VReg4S(21);
    const VReg4S v_acc12 = VReg4S(22);
    const VReg4S v_acc13 = VReg4S(23);
    const VReg4S v_acc20 = VReg4S(24);
    const VReg4S v_acc21 = VReg4S(25);
    const VReg4S v_acc22 = VReg4S(26);
    const VReg4S v_acc23 = VReg4S(27);
    const VReg4S v_acc30 = VReg4S(28);
    const VReg4S v_acc31 = VReg4S(29);
    const VReg4S v_acc32 = VReg4S(30);
    const VReg4S v_acc33 = VReg4S(31);

    const VReg16B v_acc00b(v_acc00.getIdx());
    const VReg16B v_acc01b(v_acc01.getIdx());
    const VReg16B v_acc02b(v_acc02.getIdx());
    const VReg16B v_acc03b(v_acc03.getIdx());
    const VReg16B v_acc10b(v_acc10.getIdx());
    const VReg16B v_acc11b(v_acc11.getIdx());
    const VReg16B v_acc12b(v_acc12.getIdx());
    const VReg16B v_acc13b(v_acc13.getIdx());
    const VReg16B v_acc20b(v_acc20.getIdx());
    const VReg16B v_acc21b(v_acc21.getIdx());
    const VReg16B v_acc22b(v_acc22.getIdx());
    const VReg16B v_acc23b(v_acc23.getIdx());
    const VReg16B v_acc30b(v_acc30.getIdx());
    const VReg16B v_acc31b(v_acc31.getIdx());
    const VReg16B v_acc32b(v_acc32.getIdx());
    const VReg16B v_acc33b(v_acc33.getIdx());

    const WReg reg_mask = w10;
    mov(reg_mask, 0x80);
    dup(v_mask, reg_mask);

    eor(v_acc00b, v_acc00b, v_acc00b);
    eor(v_acc01b, v_acc01b, v_acc01b);
    eor(v_acc02b, v_acc02b, v_acc02b);
    eor(v_acc03b, v_acc03b, v_acc03b);
    eor(v_acc10b, v_acc10b, v_acc10b);
    eor(v_acc11b, v_acc11b, v_acc11b);
    eor(v_acc12b, v_acc12b, v_acc12b);
    eor(v_acc13b, v_acc13b, v_acc13b);
    eor(v_acc20b, v_acc20b, v_acc20b);
    eor(v_acc21b, v_acc21b, v_acc21b);
    eor(v_acc22b, v_acc22b, v_acc22b);
    eor(v_acc23b, v_acc23b, v_acc23b);
    eor(v_acc30b, v_acc30b, v_acc30b);
    eor(v_acc31b, v_acc31b, v_acc31b);
    eor(v_acc32b, v_acc32b, v_acc32b);
    eor(v_acc33b, v_acc33b, v_acc33b);

    ldr(reg_src0, ptr(reg_srcs));
    ldr(reg_src1, ptr(reg_srcs, 8));
    ldr(reg_src2, ptr(reg_srcs, 16));
    ldr(reg_src3, ptr(reg_srcs, 24));

    mov(reg_c0, reg_dst);
    add(reg_c1, reg_c0, reg_ldc);
    add(reg_c2, reg_c1, reg_ldc);
    add(reg_c3, reg_c2, reg_ldc);

    Label loop32;
    Label loop16;
    Label reduce_store;
    Label tail_loop;
    Label done;

    cmp(reg_k, 32);
    b(LT, loop16);

    L(loop32);
    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(v_src2, ptr(reg_src2));
    ld1(v_src3, ptr(reg_src3));
    ld1(v_wlist, ptr(reg_wei));
    eor(v_src0, v_src0, v_mask);
    eor(v_src1, v_src1, v_mask);
    eor(v_src2, v_src2, v_mask);
    eor(v_src3, v_src3, v_mask);
    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
    add(reg_src2, reg_src2, 16);
    add(reg_src3, reg_src3, 16);
    add(reg_wei, reg_wei, 64);

    sdot(v_acc00, v_src0, v_w0);
    sdot(v_acc01, v_src0, v_w1);
    sdot(v_acc02, v_src0, v_w2);
    sdot(v_acc03, v_src0, v_w3);

    sdot(v_acc10, v_src1, v_w0);
    sdot(v_acc11, v_src1, v_w1);
    sdot(v_acc12, v_src1, v_w2);
    sdot(v_acc13, v_src1, v_w3);

    sdot(v_acc20, v_src2, v_w0);
    sdot(v_acc21, v_src2, v_w1);
    sdot(v_acc22, v_src2, v_w2);
    sdot(v_acc23, v_src2, v_w3);

    sdot(v_acc30, v_src3, v_w0);
    sdot(v_acc31, v_src3, v_w1);
    sdot(v_acc32, v_src3, v_w2);
    sdot(v_acc33, v_src3, v_w3);

    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(v_src2, ptr(reg_src2));
    ld1(v_src3, ptr(reg_src3));
    ld1(v_wlist, ptr(reg_wei));
    eor(v_src0, v_src0, v_mask);
    eor(v_src1, v_src1, v_mask);
    eor(v_src2, v_src2, v_mask);
    eor(v_src3, v_src3, v_mask);
    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
    add(reg_src2, reg_src2, 16);
    add(reg_src3, reg_src3, 16);
    add(reg_wei, reg_wei, 64);

    sdot(v_acc00, v_src0, v_w0);
    sdot(v_acc01, v_src0, v_w1);
    sdot(v_acc02, v_src0, v_w2);
    sdot(v_acc03, v_src0, v_w3);

    sdot(v_acc10, v_src1, v_w0);
    sdot(v_acc11, v_src1, v_w1);
    sdot(v_acc12, v_src1, v_w2);
    sdot(v_acc13, v_src1, v_w3);

    sdot(v_acc20, v_src2, v_w0);
    sdot(v_acc21, v_src2, v_w1);
    sdot(v_acc22, v_src2, v_w2);
    sdot(v_acc23, v_src2, v_w3);

    sdot(v_acc30, v_src3, v_w0);
    sdot(v_acc31, v_src3, v_w1);
    sdot(v_acc32, v_src3, v_w2);
    sdot(v_acc33, v_src3, v_w3);

    sub(reg_k, reg_k, 32);
    cmp(reg_k, 32);
    b(GE, loop32);

    L(loop16);
    cmp(reg_k, 16);
    b(LT, reduce_store);
    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(v_src2, ptr(reg_src2));
    ld1(v_src3, ptr(reg_src3));
    ld1(v_wlist, ptr(reg_wei));
    eor(v_src0, v_src0, v_mask);
    eor(v_src1, v_src1, v_mask);
    eor(v_src2, v_src2, v_mask);
    eor(v_src3, v_src3, v_mask);
    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
    add(reg_src2, reg_src2, 16);
    add(reg_src3, reg_src3, 16);
    add(reg_wei, reg_wei, 64);

    sdot(v_acc00, v_src0, v_w0);
    sdot(v_acc01, v_src0, v_w1);
    sdot(v_acc02, v_src0, v_w2);
    sdot(v_acc03, v_src0, v_w3);

    sdot(v_acc10, v_src1, v_w0);
    sdot(v_acc11, v_src1, v_w1);
    sdot(v_acc12, v_src1, v_w2);
    sdot(v_acc13, v_src1, v_w3);

    sdot(v_acc20, v_src2, v_w0);
    sdot(v_acc21, v_src2, v_w1);
    sdot(v_acc22, v_src2, v_w2);
    sdot(v_acc23, v_src2, v_w3);

    sdot(v_acc30, v_src3, v_w0);
    sdot(v_acc31, v_src3, v_w1);
    sdot(v_acc32, v_src3, v_w2);
    sdot(v_acc33, v_src3, v_w3);

    sub(reg_k, reg_k, 16);

    L(reduce_store);
    const VReg v_tmp(0);
    const SReg s_tmp(v_tmp.getIdx());
    const WReg reg_tmp = w9;
    const WReg reg_c_acc = w0;
    const WReg reg_accum_w(reg_accum.getIdx());

    addv(s_tmp, v_acc00);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0));
    addv(s_tmp, v_acc01);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 4));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 4));
    addv(s_tmp, v_acc02);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 8));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 8));
    addv(s_tmp, v_acc03);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 12));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 12));

    addv(s_tmp, v_acc10);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1));
    addv(s_tmp, v_acc11);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 4));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 4));
    addv(s_tmp, v_acc12);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 8));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 8));
    addv(s_tmp, v_acc13);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 12));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 12));

    addv(s_tmp, v_acc20);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c2));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c2));
    addv(s_tmp, v_acc21);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c2, 4));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 4));
    addv(s_tmp, v_acc22);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c2, 8));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 8));
    addv(s_tmp, v_acc23);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c2, 12));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 12));

    addv(s_tmp, v_acc30);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c3));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c3));
    addv(s_tmp, v_acc31);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c3, 4));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 4));
    addv(s_tmp, v_acc32);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c3, 8));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 8));
    addv(s_tmp, v_acc33);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c3, 12));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 12));

    cbz(reg_k, done);

    const WReg reg_a0 = w22;
    const WReg reg_a1 = w23;
    const WReg reg_a2 = w24;
    const WReg reg_a3 = w25;
    const WReg reg_b0 = w26;
    const WReg reg_b1 = w27;
    const WReg reg_b2 = w28;
    const WReg reg_b3 = w8;

    L(tail_loop);
    ldrb(reg_a0, ptr(reg_src0));
    ldrb(reg_a1, ptr(reg_src1));
    ldrb(reg_a2, ptr(reg_src2));
    ldrb(reg_a3, ptr(reg_src3));
    sub(reg_a0, reg_a0, 128);
    sub(reg_a1, reg_a1, 128);
    sub(reg_a2, reg_a2, 128);
    sub(reg_a3, reg_a3, 128);
    ldrsb(reg_b0, ptr(reg_wei));
    ldrsb(reg_b1, ptr(reg_wei, 1));
    ldrsb(reg_b2, ptr(reg_wei, 2));
    ldrsb(reg_b3, ptr(reg_wei, 3));

    add(reg_src0, reg_src0, 1);
    add(reg_src1, reg_src1, 1);
    add(reg_src2, reg_src2, 1);
    add(reg_src3, reg_src3, 1);
    add(reg_wei, reg_wei, 4);

    ldr(reg_tmp, ptr(reg_c0));
    madd(reg_tmp, reg_a0, reg_b0, reg_tmp);
    str(reg_tmp, ptr(reg_c0));
    ldr(reg_tmp, ptr(reg_c0, 4));
    madd(reg_tmp, reg_a0, reg_b1, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 4));
    ldr(reg_tmp, ptr(reg_c0, 8));
    madd(reg_tmp, reg_a0, reg_b2, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 8));
    ldr(reg_tmp, ptr(reg_c0, 12));
    madd(reg_tmp, reg_a0, reg_b3, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 12));

    ldr(reg_tmp, ptr(reg_c1));
    madd(reg_tmp, reg_a1, reg_b0, reg_tmp);
    str(reg_tmp, ptr(reg_c1));
    ldr(reg_tmp, ptr(reg_c1, 4));
    madd(reg_tmp, reg_a1, reg_b1, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 4));
    ldr(reg_tmp, ptr(reg_c1, 8));
    madd(reg_tmp, reg_a1, reg_b2, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 8));
    ldr(reg_tmp, ptr(reg_c1, 12));
    madd(reg_tmp, reg_a1, reg_b3, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 12));

    ldr(reg_tmp, ptr(reg_c2));
    madd(reg_tmp, reg_a2, reg_b0, reg_tmp);
    str(reg_tmp, ptr(reg_c2));
    ldr(reg_tmp, ptr(reg_c2, 4));
    madd(reg_tmp, reg_a2, reg_b1, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 4));
    ldr(reg_tmp, ptr(reg_c2, 8));
    madd(reg_tmp, reg_a2, reg_b2, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 8));
    ldr(reg_tmp, ptr(reg_c2, 12));
    madd(reg_tmp, reg_a2, reg_b3, reg_tmp);
    str(reg_tmp, ptr(reg_c2, 12));

    ldr(reg_tmp, ptr(reg_c3));
    madd(reg_tmp, reg_a3, reg_b0, reg_tmp);
    str(reg_tmp, ptr(reg_c3));
    ldr(reg_tmp, ptr(reg_c3, 4));
    madd(reg_tmp, reg_a3, reg_b1, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 4));
    ldr(reg_tmp, ptr(reg_c3, 8));
    madd(reg_tmp, reg_a3, reg_b2, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 8));
    ldr(reg_tmp, ptr(reg_c3, 12));
    madd(reg_tmp, reg_a3, reg_b3, reg_tmp);
    str(reg_tmp, ptr(reg_c3, 12));

    subs(reg_k, reg_k, 1);
    b(NE, tail_loop);

    L(done);
    postamble();
}

jit_int8_brgemm_kernel_2x8_dot::jit_int8_brgemm_kernel_2x8_dot() : jit_generator() {}

void jit_int8_brgemm_kernel_2x8_dot::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_2x8_dot::generate() {
    preamble();

    const XReg reg_srcs = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_ldb = abi_param5;
    const XReg reg_ldc = abi_param6;
    const XReg reg_accum = abi_param7;

    const XReg reg_src0 = x10;
    const XReg reg_src1 = x11;

    const XReg reg_w0 = x12;
    const XReg reg_w1 = x13;
    const XReg reg_w2 = x14;
    const XReg reg_w3 = x15;
    const XReg reg_w4 = x16;
    const XReg reg_w5 = x17;
    const XReg reg_w6 = x18;
    const XReg reg_w7 = x19;

    const XReg reg_c0 = x20;
    const XReg reg_c1 = x21;

    const VReg16B v_src0 = VReg16B(0);
    const VReg16B v_src1 = VReg16B(1);
    const VReg16B v_w0 = VReg16B(2);
    const VReg16B v_w1 = VReg16B(3);
    const VReg16B v_w2 = VReg16B(4);
    const VReg16B v_w3 = VReg16B(5);
    const VReg16B v_w4 = VReg16B(6);
    const VReg16B v_w5 = VReg16B(7);
    const VReg16B v_w6 = VReg16B(8);
    const VReg16B v_w7 = VReg16B(9);

    const VReg4S v_acc00 = VReg4S(16);
    const VReg4S v_acc01 = VReg4S(17);
    const VReg4S v_acc02 = VReg4S(18);
    const VReg4S v_acc03 = VReg4S(19);
    const VReg4S v_acc04 = VReg4S(20);
    const VReg4S v_acc05 = VReg4S(21);
    const VReg4S v_acc06 = VReg4S(22);
    const VReg4S v_acc07 = VReg4S(23);
    const VReg4S v_acc10 = VReg4S(24);
    const VReg4S v_acc11 = VReg4S(25);
    const VReg4S v_acc12 = VReg4S(26);
    const VReg4S v_acc13 = VReg4S(27);
    const VReg4S v_acc14 = VReg4S(28);
    const VReg4S v_acc15 = VReg4S(29);
    const VReg4S v_acc16 = VReg4S(30);
    const VReg4S v_acc17 = VReg4S(31);

    const VReg16B v_acc00b(v_acc00.getIdx());
    const VReg16B v_acc01b(v_acc01.getIdx());
    const VReg16B v_acc02b(v_acc02.getIdx());
    const VReg16B v_acc03b(v_acc03.getIdx());
    const VReg16B v_acc04b(v_acc04.getIdx());
    const VReg16B v_acc05b(v_acc05.getIdx());
    const VReg16B v_acc06b(v_acc06.getIdx());
    const VReg16B v_acc07b(v_acc07.getIdx());
    const VReg16B v_acc10b(v_acc10.getIdx());
    const VReg16B v_acc11b(v_acc11.getIdx());
    const VReg16B v_acc12b(v_acc12.getIdx());
    const VReg16B v_acc13b(v_acc13.getIdx());
    const VReg16B v_acc14b(v_acc14.getIdx());
    const VReg16B v_acc15b(v_acc15.getIdx());
    const VReg16B v_acc16b(v_acc16.getIdx());
    const VReg16B v_acc17b(v_acc17.getIdx());

    eor(v_acc00b, v_acc00b, v_acc00b);
    eor(v_acc01b, v_acc01b, v_acc01b);
    eor(v_acc02b, v_acc02b, v_acc02b);
    eor(v_acc03b, v_acc03b, v_acc03b);
    eor(v_acc04b, v_acc04b, v_acc04b);
    eor(v_acc05b, v_acc05b, v_acc05b);
    eor(v_acc06b, v_acc06b, v_acc06b);
    eor(v_acc07b, v_acc07b, v_acc07b);
    eor(v_acc10b, v_acc10b, v_acc10b);
    eor(v_acc11b, v_acc11b, v_acc11b);
    eor(v_acc12b, v_acc12b, v_acc12b);
    eor(v_acc13b, v_acc13b, v_acc13b);
    eor(v_acc14b, v_acc14b, v_acc14b);
    eor(v_acc15b, v_acc15b, v_acc15b);
    eor(v_acc16b, v_acc16b, v_acc16b);
    eor(v_acc17b, v_acc17b, v_acc17b);

    ldr(reg_src0, ptr(reg_srcs));
    ldr(reg_src1, ptr(reg_srcs, 8));

    mov(reg_w0, reg_wei);
    add(reg_w1, reg_w0, reg_ldb);
    add(reg_w2, reg_w1, reg_ldb);
    add(reg_w3, reg_w2, reg_ldb);
    add(reg_w4, reg_w3, reg_ldb);
    add(reg_w5, reg_w4, reg_ldb);
    add(reg_w6, reg_w5, reg_ldb);
    add(reg_w7, reg_w6, reg_ldb);

    mov(reg_c0, reg_dst);
    add(reg_c1, reg_c0, reg_ldc);

    Label loop32;
    Label loop16;
    Label reduce_store;
    Label tail_loop;
    Label done;

    cmp(reg_k, 32);
    b(LT, loop16);

    L(loop32);
    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(v_w0, ptr(reg_w0));
    ld1(v_w1, ptr(reg_w1));
    ld1(v_w2, ptr(reg_w2));
    ld1(v_w3, ptr(reg_w3));
    ld1(v_w4, ptr(reg_w4));
    ld1(v_w5, ptr(reg_w5));
    ld1(v_w6, ptr(reg_w6));
    ld1(v_w7, ptr(reg_w7));

    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
    add(reg_w0, reg_w0, 16);
    add(reg_w1, reg_w1, 16);
    add(reg_w2, reg_w2, 16);
    add(reg_w3, reg_w3, 16);
    add(reg_w4, reg_w4, 16);
    add(reg_w5, reg_w5, 16);
    add(reg_w6, reg_w6, 16);
    add(reg_w7, reg_w7, 16);

    sdot(v_acc00, v_src0, v_w0);
    sdot(v_acc01, v_src0, v_w1);
    sdot(v_acc02, v_src0, v_w2);
    sdot(v_acc03, v_src0, v_w3);
    sdot(v_acc04, v_src0, v_w4);
    sdot(v_acc05, v_src0, v_w5);
    sdot(v_acc06, v_src0, v_w6);
    sdot(v_acc07, v_src0, v_w7);

    sdot(v_acc10, v_src1, v_w0);
    sdot(v_acc11, v_src1, v_w1);
    sdot(v_acc12, v_src1, v_w2);
    sdot(v_acc13, v_src1, v_w3);
    sdot(v_acc14, v_src1, v_w4);
    sdot(v_acc15, v_src1, v_w5);
    sdot(v_acc16, v_src1, v_w6);
    sdot(v_acc17, v_src1, v_w7);

    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(v_w0, ptr(reg_w0));
    ld1(v_w1, ptr(reg_w1));
    ld1(v_w2, ptr(reg_w2));
    ld1(v_w3, ptr(reg_w3));
    ld1(v_w4, ptr(reg_w4));
    ld1(v_w5, ptr(reg_w5));
    ld1(v_w6, ptr(reg_w6));
    ld1(v_w7, ptr(reg_w7));

    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
    add(reg_w0, reg_w0, 16);
    add(reg_w1, reg_w1, 16);
    add(reg_w2, reg_w2, 16);
    add(reg_w3, reg_w3, 16);
    add(reg_w4, reg_w4, 16);
    add(reg_w5, reg_w5, 16);
    add(reg_w6, reg_w6, 16);
    add(reg_w7, reg_w7, 16);

    sdot(v_acc00, v_src0, v_w0);
    sdot(v_acc01, v_src0, v_w1);
    sdot(v_acc02, v_src0, v_w2);
    sdot(v_acc03, v_src0, v_w3);
    sdot(v_acc04, v_src0, v_w4);
    sdot(v_acc05, v_src0, v_w5);
    sdot(v_acc06, v_src0, v_w6);
    sdot(v_acc07, v_src0, v_w7);

    sdot(v_acc10, v_src1, v_w0);
    sdot(v_acc11, v_src1, v_w1);
    sdot(v_acc12, v_src1, v_w2);
    sdot(v_acc13, v_src1, v_w3);
    sdot(v_acc14, v_src1, v_w4);
    sdot(v_acc15, v_src1, v_w5);
    sdot(v_acc16, v_src1, v_w6);
    sdot(v_acc17, v_src1, v_w7);

    sub(reg_k, reg_k, 32);
    cmp(reg_k, 32);
    b(GE, loop32);

    L(loop16);
    cmp(reg_k, 16);
    b(LT, reduce_store);
    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(v_w0, ptr(reg_w0));
    ld1(v_w1, ptr(reg_w1));
    ld1(v_w2, ptr(reg_w2));
    ld1(v_w3, ptr(reg_w3));
    ld1(v_w4, ptr(reg_w4));
    ld1(v_w5, ptr(reg_w5));
    ld1(v_w6, ptr(reg_w6));
    ld1(v_w7, ptr(reg_w7));

    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
    add(reg_w0, reg_w0, 16);
    add(reg_w1, reg_w1, 16);
    add(reg_w2, reg_w2, 16);
    add(reg_w3, reg_w3, 16);
    add(reg_w4, reg_w4, 16);
    add(reg_w5, reg_w5, 16);
    add(reg_w6, reg_w6, 16);
    add(reg_w7, reg_w7, 16);

    sdot(v_acc00, v_src0, v_w0);
    sdot(v_acc01, v_src0, v_w1);
    sdot(v_acc02, v_src0, v_w2);
    sdot(v_acc03, v_src0, v_w3);
    sdot(v_acc04, v_src0, v_w4);
    sdot(v_acc05, v_src0, v_w5);
    sdot(v_acc06, v_src0, v_w6);
    sdot(v_acc07, v_src0, v_w7);

    sdot(v_acc10, v_src1, v_w0);
    sdot(v_acc11, v_src1, v_w1);
    sdot(v_acc12, v_src1, v_w2);
    sdot(v_acc13, v_src1, v_w3);
    sdot(v_acc14, v_src1, v_w4);
    sdot(v_acc15, v_src1, v_w5);
    sdot(v_acc16, v_src1, v_w6);
    sdot(v_acc17, v_src1, v_w7);

    sub(reg_k, reg_k, 16);

    L(reduce_store);
    const VReg v_tmp(0);
    const SReg s_tmp(v_tmp.getIdx());
    const WReg reg_tmp = w9;
    const WReg reg_c_acc = w0;
    const WReg reg_accum_w(reg_accum.getIdx());

    addv(s_tmp, v_acc00);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0));
    addv(s_tmp, v_acc01);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 4));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 4));
    addv(s_tmp, v_acc02);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 8));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 8));
    addv(s_tmp, v_acc03);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 12));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 12));
    addv(s_tmp, v_acc04);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 16));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 16));
    addv(s_tmp, v_acc05);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 20));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 20));
    addv(s_tmp, v_acc06);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 24));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 24));
    addv(s_tmp, v_acc07);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 28));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 28));

    addv(s_tmp, v_acc10);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1));
    addv(s_tmp, v_acc11);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 4));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 4));
    addv(s_tmp, v_acc12);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 8));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 8));
    addv(s_tmp, v_acc13);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 12));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 12));
    addv(s_tmp, v_acc14);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 16));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 16));
    addv(s_tmp, v_acc15);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 20));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 20));
    addv(s_tmp, v_acc16);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 24));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 24));
    addv(s_tmp, v_acc17);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 28));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 28));

    cbz(reg_k, done);

    const WReg reg_a0 = w22;
    const WReg reg_a1 = w23;
    const WReg reg_b = w24;

    L(tail_loop);
    ldrsb(reg_a0, ptr(reg_src0));
    ldrsb(reg_a1, ptr(reg_src1));

    ldrsb(reg_b, ptr(reg_w0));
    ldr(reg_tmp, ptr(reg_c0));
    madd(reg_tmp, reg_a0, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c0));
    ldr(reg_tmp, ptr(reg_c1));
    madd(reg_tmp, reg_a1, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c1));

    ldrsb(reg_b, ptr(reg_w1));
    ldr(reg_tmp, ptr(reg_c0, 4));
    madd(reg_tmp, reg_a0, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 4));
    ldr(reg_tmp, ptr(reg_c1, 4));
    madd(reg_tmp, reg_a1, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 4));

    ldrsb(reg_b, ptr(reg_w2));
    ldr(reg_tmp, ptr(reg_c0, 8));
    madd(reg_tmp, reg_a0, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 8));
    ldr(reg_tmp, ptr(reg_c1, 8));
    madd(reg_tmp, reg_a1, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 8));

    ldrsb(reg_b, ptr(reg_w3));
    ldr(reg_tmp, ptr(reg_c0, 12));
    madd(reg_tmp, reg_a0, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 12));
    ldr(reg_tmp, ptr(reg_c1, 12));
    madd(reg_tmp, reg_a1, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 12));

    ldrsb(reg_b, ptr(reg_w4));
    ldr(reg_tmp, ptr(reg_c0, 16));
    madd(reg_tmp, reg_a0, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 16));
    ldr(reg_tmp, ptr(reg_c1, 16));
    madd(reg_tmp, reg_a1, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 16));

    ldrsb(reg_b, ptr(reg_w5));
    ldr(reg_tmp, ptr(reg_c0, 20));
    madd(reg_tmp, reg_a0, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 20));
    ldr(reg_tmp, ptr(reg_c1, 20));
    madd(reg_tmp, reg_a1, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 20));

    ldrsb(reg_b, ptr(reg_w6));
    ldr(reg_tmp, ptr(reg_c0, 24));
    madd(reg_tmp, reg_a0, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 24));
    ldr(reg_tmp, ptr(reg_c1, 24));
    madd(reg_tmp, reg_a1, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 24));

    ldrsb(reg_b, ptr(reg_w7));
    ldr(reg_tmp, ptr(reg_c0, 28));
    madd(reg_tmp, reg_a0, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 28));
    ldr(reg_tmp, ptr(reg_c1, 28));
    madd(reg_tmp, reg_a1, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 28));

    add(reg_src0, reg_src0, 1);
    add(reg_src1, reg_src1, 1);
    add(reg_w0, reg_w0, 1);
    add(reg_w1, reg_w1, 1);
    add(reg_w2, reg_w2, 1);
    add(reg_w3, reg_w3, 1);
    add(reg_w4, reg_w4, 1);
    add(reg_w5, reg_w5, 1);
    add(reg_w6, reg_w6, 1);
    add(reg_w7, reg_w7, 1);

    subs(reg_k, reg_k, 1);
    b(NE, tail_loop);

    L(done);
    postamble();
}

jit_int8_brgemm_kernel_2x8_udot::jit_int8_brgemm_kernel_2x8_udot() : jit_generator() {}

void jit_int8_brgemm_kernel_2x8_udot::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_2x8_udot::generate() {
    preamble();

    const XReg reg_srcs = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_ldb = abi_param5;
    const XReg reg_ldc = abi_param6;
    const XReg reg_accum = abi_param7;

    const XReg reg_src0 = x10;
    const XReg reg_src1 = x11;

    const XReg reg_w0 = x12;
    const XReg reg_w1 = x13;
    const XReg reg_w2 = x14;
    const XReg reg_w3 = x15;
    const XReg reg_w4 = x16;
    const XReg reg_w5 = x17;
    const XReg reg_w6 = x18;
    const XReg reg_w7 = x19;

    const XReg reg_c0 = x20;
    const XReg reg_c1 = x21;

    const VReg16B v_src0 = VReg16B(0);
    const VReg16B v_src1 = VReg16B(1);
    const VReg16B v_w0 = VReg16B(2);
    const VReg16B v_w1 = VReg16B(3);
    const VReg16B v_w2 = VReg16B(4);
    const VReg16B v_w3 = VReg16B(5);
    const VReg16B v_w4 = VReg16B(6);
    const VReg16B v_w5 = VReg16B(7);
    const VReg16B v_w6 = VReg16B(8);
    const VReg16B v_w7 = VReg16B(9);
    const VReg16B v_mask = VReg16B(10);

    const VReg4S v_acc00 = VReg4S(16);
    const VReg4S v_acc01 = VReg4S(17);
    const VReg4S v_acc02 = VReg4S(18);
    const VReg4S v_acc03 = VReg4S(19);
    const VReg4S v_acc04 = VReg4S(20);
    const VReg4S v_acc05 = VReg4S(21);
    const VReg4S v_acc06 = VReg4S(22);
    const VReg4S v_acc07 = VReg4S(23);
    const VReg4S v_acc10 = VReg4S(24);
    const VReg4S v_acc11 = VReg4S(25);
    const VReg4S v_acc12 = VReg4S(26);
    const VReg4S v_acc13 = VReg4S(27);
    const VReg4S v_acc14 = VReg4S(28);
    const VReg4S v_acc15 = VReg4S(29);
    const VReg4S v_acc16 = VReg4S(30);
    const VReg4S v_acc17 = VReg4S(31);

    const VReg16B v_acc00b(v_acc00.getIdx());
    const VReg16B v_acc01b(v_acc01.getIdx());
    const VReg16B v_acc02b(v_acc02.getIdx());
    const VReg16B v_acc03b(v_acc03.getIdx());
    const VReg16B v_acc04b(v_acc04.getIdx());
    const VReg16B v_acc05b(v_acc05.getIdx());
    const VReg16B v_acc06b(v_acc06.getIdx());
    const VReg16B v_acc07b(v_acc07.getIdx());
    const VReg16B v_acc10b(v_acc10.getIdx());
    const VReg16B v_acc11b(v_acc11.getIdx());
    const VReg16B v_acc12b(v_acc12.getIdx());
    const VReg16B v_acc13b(v_acc13.getIdx());
    const VReg16B v_acc14b(v_acc14.getIdx());
    const VReg16B v_acc15b(v_acc15.getIdx());
    const VReg16B v_acc16b(v_acc16.getIdx());
    const VReg16B v_acc17b(v_acc17.getIdx());

    const WReg reg_mask = w25;
    mov(reg_mask, 0x80);
    dup(v_mask, reg_mask);

    eor(v_acc00b, v_acc00b, v_acc00b);
    eor(v_acc01b, v_acc01b, v_acc01b);
    eor(v_acc02b, v_acc02b, v_acc02b);
    eor(v_acc03b, v_acc03b, v_acc03b);
    eor(v_acc04b, v_acc04b, v_acc04b);
    eor(v_acc05b, v_acc05b, v_acc05b);
    eor(v_acc06b, v_acc06b, v_acc06b);
    eor(v_acc07b, v_acc07b, v_acc07b);
    eor(v_acc10b, v_acc10b, v_acc10b);
    eor(v_acc11b, v_acc11b, v_acc11b);
    eor(v_acc12b, v_acc12b, v_acc12b);
    eor(v_acc13b, v_acc13b, v_acc13b);
    eor(v_acc14b, v_acc14b, v_acc14b);
    eor(v_acc15b, v_acc15b, v_acc15b);
    eor(v_acc16b, v_acc16b, v_acc16b);
    eor(v_acc17b, v_acc17b, v_acc17b);

    ldr(reg_src0, ptr(reg_srcs));
    ldr(reg_src1, ptr(reg_srcs, 8));

    mov(reg_w0, reg_wei);
    add(reg_w1, reg_w0, reg_ldb);
    add(reg_w2, reg_w1, reg_ldb);
    add(reg_w3, reg_w2, reg_ldb);
    add(reg_w4, reg_w3, reg_ldb);
    add(reg_w5, reg_w4, reg_ldb);
    add(reg_w6, reg_w5, reg_ldb);
    add(reg_w7, reg_w6, reg_ldb);

    mov(reg_c0, reg_dst);
    add(reg_c1, reg_c0, reg_ldc);

    Label loop32;
    Label loop16;
    Label reduce_store;
    Label tail_loop;
    Label done;

    cmp(reg_k, 32);
    b(LT, loop16);

    L(loop32);
    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(v_w0, ptr(reg_w0));
    ld1(v_w1, ptr(reg_w1));
    ld1(v_w2, ptr(reg_w2));
    ld1(v_w3, ptr(reg_w3));
    ld1(v_w4, ptr(reg_w4));
    ld1(v_w5, ptr(reg_w5));
    ld1(v_w6, ptr(reg_w6));
    ld1(v_w7, ptr(reg_w7));

    eor(v_src0, v_src0, v_mask);
    eor(v_src1, v_src1, v_mask);

    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
    add(reg_w0, reg_w0, 16);
    add(reg_w1, reg_w1, 16);
    add(reg_w2, reg_w2, 16);
    add(reg_w3, reg_w3, 16);
    add(reg_w4, reg_w4, 16);
    add(reg_w5, reg_w5, 16);
    add(reg_w6, reg_w6, 16);
    add(reg_w7, reg_w7, 16);

    sdot(v_acc00, v_src0, v_w0);
    sdot(v_acc01, v_src0, v_w1);
    sdot(v_acc02, v_src0, v_w2);
    sdot(v_acc03, v_src0, v_w3);
    sdot(v_acc04, v_src0, v_w4);
    sdot(v_acc05, v_src0, v_w5);
    sdot(v_acc06, v_src0, v_w6);
    sdot(v_acc07, v_src0, v_w7);

    sdot(v_acc10, v_src1, v_w0);
    sdot(v_acc11, v_src1, v_w1);
    sdot(v_acc12, v_src1, v_w2);
    sdot(v_acc13, v_src1, v_w3);
    sdot(v_acc14, v_src1, v_w4);
    sdot(v_acc15, v_src1, v_w5);
    sdot(v_acc16, v_src1, v_w6);
    sdot(v_acc17, v_src1, v_w7);

    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(v_w0, ptr(reg_w0));
    ld1(v_w1, ptr(reg_w1));
    ld1(v_w2, ptr(reg_w2));
    ld1(v_w3, ptr(reg_w3));
    ld1(v_w4, ptr(reg_w4));
    ld1(v_w5, ptr(reg_w5));
    ld1(v_w6, ptr(reg_w6));
    ld1(v_w7, ptr(reg_w7));

    eor(v_src0, v_src0, v_mask);
    eor(v_src1, v_src1, v_mask);

    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
    add(reg_w0, reg_w0, 16);
    add(reg_w1, reg_w1, 16);
    add(reg_w2, reg_w2, 16);
    add(reg_w3, reg_w3, 16);
    add(reg_w4, reg_w4, 16);
    add(reg_w5, reg_w5, 16);
    add(reg_w6, reg_w6, 16);
    add(reg_w7, reg_w7, 16);

    sdot(v_acc00, v_src0, v_w0);
    sdot(v_acc01, v_src0, v_w1);
    sdot(v_acc02, v_src0, v_w2);
    sdot(v_acc03, v_src0, v_w3);
    sdot(v_acc04, v_src0, v_w4);
    sdot(v_acc05, v_src0, v_w5);
    sdot(v_acc06, v_src0, v_w6);
    sdot(v_acc07, v_src0, v_w7);

    sdot(v_acc10, v_src1, v_w0);
    sdot(v_acc11, v_src1, v_w1);
    sdot(v_acc12, v_src1, v_w2);
    sdot(v_acc13, v_src1, v_w3);
    sdot(v_acc14, v_src1, v_w4);
    sdot(v_acc15, v_src1, v_w5);
    sdot(v_acc16, v_src1, v_w6);
    sdot(v_acc17, v_src1, v_w7);

    sub(reg_k, reg_k, 32);
    cmp(reg_k, 32);
    b(GE, loop32);

    L(loop16);
    cmp(reg_k, 16);
    b(LT, reduce_store);
    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(v_w0, ptr(reg_w0));
    ld1(v_w1, ptr(reg_w1));
    ld1(v_w2, ptr(reg_w2));
    ld1(v_w3, ptr(reg_w3));
    ld1(v_w4, ptr(reg_w4));
    ld1(v_w5, ptr(reg_w5));
    ld1(v_w6, ptr(reg_w6));
    ld1(v_w7, ptr(reg_w7));

    eor(v_src0, v_src0, v_mask);
    eor(v_src1, v_src1, v_mask);

    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
    add(reg_w0, reg_w0, 16);
    add(reg_w1, reg_w1, 16);
    add(reg_w2, reg_w2, 16);
    add(reg_w3, reg_w3, 16);
    add(reg_w4, reg_w4, 16);
    add(reg_w5, reg_w5, 16);
    add(reg_w6, reg_w6, 16);
    add(reg_w7, reg_w7, 16);

    sdot(v_acc00, v_src0, v_w0);
    sdot(v_acc01, v_src0, v_w1);
    sdot(v_acc02, v_src0, v_w2);
    sdot(v_acc03, v_src0, v_w3);
    sdot(v_acc04, v_src0, v_w4);
    sdot(v_acc05, v_src0, v_w5);
    sdot(v_acc06, v_src0, v_w6);
    sdot(v_acc07, v_src0, v_w7);

    sdot(v_acc10, v_src1, v_w0);
    sdot(v_acc11, v_src1, v_w1);
    sdot(v_acc12, v_src1, v_w2);
    sdot(v_acc13, v_src1, v_w3);
    sdot(v_acc14, v_src1, v_w4);
    sdot(v_acc15, v_src1, v_w5);
    sdot(v_acc16, v_src1, v_w6);
    sdot(v_acc17, v_src1, v_w7);

    sub(reg_k, reg_k, 16);

    L(reduce_store);
    const VReg v_tmp(0);
    const SReg s_tmp(v_tmp.getIdx());
    const WReg reg_tmp = w9;
    const WReg reg_c_acc = w0;
    const WReg reg_accum_w(reg_accum.getIdx());

    addv(s_tmp, v_acc00);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0));
    addv(s_tmp, v_acc01);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 4));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 4));
    addv(s_tmp, v_acc02);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 8));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 8));
    addv(s_tmp, v_acc03);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 12));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 12));
    addv(s_tmp, v_acc04);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 16));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 16));
    addv(s_tmp, v_acc05);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 20));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 20));
    addv(s_tmp, v_acc06);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 24));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 24));
    addv(s_tmp, v_acc07);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c0, 28));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 28));

    addv(s_tmp, v_acc10);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1));
    addv(s_tmp, v_acc11);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 4));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 4));
    addv(s_tmp, v_acc12);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 8));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 8));
    addv(s_tmp, v_acc13);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 12));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 12));
    addv(s_tmp, v_acc14);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 16));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 16));
    addv(s_tmp, v_acc15);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 20));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 20));
    addv(s_tmp, v_acc16);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 24));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 24));
    addv(s_tmp, v_acc17);
    umov(reg_tmp, v_tmp.s[0]);
    ldr(reg_c_acc, ptr(reg_c1, 28));
    madd(reg_tmp, reg_accum_w, reg_c_acc, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 28));

    cbz(reg_k, done);

    const WReg reg_a0 = w22;
    const WReg reg_a1 = w23;
    const WReg reg_b = w24;

    L(tail_loop);
    ldrb(reg_a0, ptr(reg_src0));
    ldrb(reg_a1, ptr(reg_src1));
    sub(reg_a0, reg_a0, 128);
    sub(reg_a1, reg_a1, 128);

    ldrsb(reg_b, ptr(reg_w0));
    ldr(reg_tmp, ptr(reg_c0));
    madd(reg_tmp, reg_a0, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c0));
    ldr(reg_tmp, ptr(reg_c1));
    madd(reg_tmp, reg_a1, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c1));

    ldrsb(reg_b, ptr(reg_w1));
    ldr(reg_tmp, ptr(reg_c0, 4));
    madd(reg_tmp, reg_a0, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 4));
    ldr(reg_tmp, ptr(reg_c1, 4));
    madd(reg_tmp, reg_a1, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 4));

    ldrsb(reg_b, ptr(reg_w2));
    ldr(reg_tmp, ptr(reg_c0, 8));
    madd(reg_tmp, reg_a0, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 8));
    ldr(reg_tmp, ptr(reg_c1, 8));
    madd(reg_tmp, reg_a1, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 8));

    ldrsb(reg_b, ptr(reg_w3));
    ldr(reg_tmp, ptr(reg_c0, 12));
    madd(reg_tmp, reg_a0, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 12));
    ldr(reg_tmp, ptr(reg_c1, 12));
    madd(reg_tmp, reg_a1, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 12));

    ldrsb(reg_b, ptr(reg_w4));
    ldr(reg_tmp, ptr(reg_c0, 16));
    madd(reg_tmp, reg_a0, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 16));
    ldr(reg_tmp, ptr(reg_c1, 16));
    madd(reg_tmp, reg_a1, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 16));

    ldrsb(reg_b, ptr(reg_w5));
    ldr(reg_tmp, ptr(reg_c0, 20));
    madd(reg_tmp, reg_a0, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 20));
    ldr(reg_tmp, ptr(reg_c1, 20));
    madd(reg_tmp, reg_a1, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 20));

    ldrsb(reg_b, ptr(reg_w6));
    ldr(reg_tmp, ptr(reg_c0, 24));
    madd(reg_tmp, reg_a0, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 24));
    ldr(reg_tmp, ptr(reg_c1, 24));
    madd(reg_tmp, reg_a1, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 24));

    ldrsb(reg_b, ptr(reg_w7));
    ldr(reg_tmp, ptr(reg_c0, 28));
    madd(reg_tmp, reg_a0, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 28));
    ldr(reg_tmp, ptr(reg_c1, 28));
    madd(reg_tmp, reg_a1, reg_b, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 28));

    add(reg_src0, reg_src0, 1);
    add(reg_src1, reg_src1, 1);
    add(reg_w0, reg_w0, 1);
    add(reg_w1, reg_w1, 1);
    add(reg_w2, reg_w2, 1);
    add(reg_w3, reg_w3, 1);
    add(reg_w4, reg_w4, 1);
    add(reg_w5, reg_w5, 1);
    add(reg_w6, reg_w6, 1);
    add(reg_w7, reg_w7, 1);

    subs(reg_k, reg_k, 1);
    b(NE, tail_loop);

    L(done);
    postamble();
}

}  // namespace ov::intel_cpu::aarch64
