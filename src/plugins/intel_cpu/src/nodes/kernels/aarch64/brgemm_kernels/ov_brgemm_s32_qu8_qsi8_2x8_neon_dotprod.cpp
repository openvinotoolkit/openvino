// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/aarch64/brgemm_kernels/int8_brgemm_kernels.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_gen.h>

#include "utils/cpu_utils.hpp"

namespace ov::intel_cpu::aarch64 {

using namespace Xbyak_aarch64;
using namespace dnnl::impl::cpu::aarch64;

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

