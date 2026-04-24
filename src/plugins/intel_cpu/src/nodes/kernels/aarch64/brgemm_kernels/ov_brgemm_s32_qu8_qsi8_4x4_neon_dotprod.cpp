// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/aarch64/brgemm_kernels/int8_brgemm_kernels.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_gen.h>

#include "utils/cpu_utils.hpp"

namespace ov::intel_cpu::aarch64 {

using namespace Xbyak_aarch64;
using namespace dnnl::impl::cpu::aarch64;

jit_int8_brgemm_kernel_4x4_udot::jit_int8_brgemm_kernel_4x4_udot() : jit_generator() {}

void jit_int8_brgemm_kernel_4x4_udot::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_4x4_udot::generate() {
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

}  // namespace ov::intel_cpu::aarch64
