// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/aarch64/brgemm_kernels/int8_brgemm_kernels.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_gen.h>

#include "utils/cpu_utils.hpp"

namespace ov::intel_cpu::aarch64 {

using namespace Xbyak_aarch64;
using namespace dnnl::impl::cpu::aarch64;

jit_int8_brgemm_kernel_2x8_dot_packed::jit_int8_brgemm_kernel_2x8_dot_packed() : jit_generator() {}

void jit_int8_brgemm_kernel_2x8_dot_packed::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_2x8_dot_packed::generate() {
    preamble();

    const XReg reg_srcs = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_ldc = abi_param6;
    const XReg reg_accum = abi_param7;

    const XReg reg_src0 = x10;
    const XReg reg_src1 = x11;

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

    auto prefetch_next = [&]() {
        prfm(PLDL1KEEP, ptr(reg_wei, 256));
        prfm(PLDL1KEEP, ptr(reg_src0, 64));
        prfm(PLDL1KEEP, ptr(reg_src1, 64));
    };

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

    mov(reg_c0, reg_dst);
    add(reg_c1, reg_c0, reg_ldc);

    Label loop64;
    Label check32;
    Label loop32;
    Label loop16;
    Label reduce_store;
    Label tail_loop;
    Label done;

    cmp(reg_k, 64);
    b(LT, check32);

    L(loop64);
    prefetch_next();
    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(VReg16BList(v_w0, v_w3), post_ptr(reg_wei, 64));
    ld1(VReg16BList(v_w4, v_w7), post_ptr(reg_wei, 64));
    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
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
    ld1(VReg16BList(v_w0, v_w3), post_ptr(reg_wei, 64));
    ld1(VReg16BList(v_w4, v_w7), post_ptr(reg_wei, 64));
    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
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
    ld1(VReg16BList(v_w0, v_w3), post_ptr(reg_wei, 64));
    ld1(VReg16BList(v_w4, v_w7), post_ptr(reg_wei, 64));
    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
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
    ld1(VReg16BList(v_w0, v_w3), post_ptr(reg_wei, 64));
    ld1(VReg16BList(v_w4, v_w7), post_ptr(reg_wei, 64));
    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
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

    sub(reg_k, reg_k, 64);
    cmp(reg_k, 64);
    b(GE, loop64);

    L(check32);
    cmp(reg_k, 32);
    b(LT, loop16);

    L(loop32);
    prefetch_next();
    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(VReg16BList(v_w0, v_w3), post_ptr(reg_wei, 64));
    ld1(VReg16BList(v_w4, v_w7), post_ptr(reg_wei, 64));
    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
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
    ld1(VReg16BList(v_w0, v_w3), post_ptr(reg_wei, 64));
    ld1(VReg16BList(v_w4, v_w7), post_ptr(reg_wei, 64));
    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
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
    prefetch_next();
    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(VReg16BList(v_w0, v_w3), post_ptr(reg_wei, 64));
    ld1(VReg16BList(v_w4, v_w7), post_ptr(reg_wei, 64));
    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
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
    const WReg reg_b0 = w14;
    const WReg reg_b1 = w15;
    const WReg reg_b2 = w16;
    const WReg reg_b3 = w17;
    const WReg reg_b4 = w18;
    const WReg reg_b5 = w19;
    const WReg reg_b6 = w24;
    const WReg reg_b7 = w25;

    L(tail_loop);
    ldrsb(reg_a0, ptr(reg_src0));
    ldrsb(reg_a1, ptr(reg_src1));

    ldrsb(reg_b0, ptr(reg_wei));
    ldrsb(reg_b1, ptr(reg_wei, 1));
    ldrsb(reg_b2, ptr(reg_wei, 2));
    ldrsb(reg_b3, ptr(reg_wei, 3));
    ldrsb(reg_b4, ptr(reg_wei, 4));
    ldrsb(reg_b5, ptr(reg_wei, 5));
    ldrsb(reg_b6, ptr(reg_wei, 6));
    ldrsb(reg_b7, ptr(reg_wei, 7));

    ldr(reg_tmp, ptr(reg_c0));
    madd(reg_tmp, reg_a0, reg_b0, reg_tmp);
    str(reg_tmp, ptr(reg_c0));
    ldr(reg_tmp, ptr(reg_c1));
    madd(reg_tmp, reg_a1, reg_b0, reg_tmp);
    str(reg_tmp, ptr(reg_c1));

    ldr(reg_tmp, ptr(reg_c0, 4));
    madd(reg_tmp, reg_a0, reg_b1, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 4));
    ldr(reg_tmp, ptr(reg_c1, 4));
    madd(reg_tmp, reg_a1, reg_b1, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 4));

    ldr(reg_tmp, ptr(reg_c0, 8));
    madd(reg_tmp, reg_a0, reg_b2, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 8));
    ldr(reg_tmp, ptr(reg_c1, 8));
    madd(reg_tmp, reg_a1, reg_b2, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 8));

    ldr(reg_tmp, ptr(reg_c0, 12));
    madd(reg_tmp, reg_a0, reg_b3, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 12));
    ldr(reg_tmp, ptr(reg_c1, 12));
    madd(reg_tmp, reg_a1, reg_b3, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 12));

    ldr(reg_tmp, ptr(reg_c0, 16));
    madd(reg_tmp, reg_a0, reg_b4, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 16));
    ldr(reg_tmp, ptr(reg_c1, 16));
    madd(reg_tmp, reg_a1, reg_b4, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 16));

    ldr(reg_tmp, ptr(reg_c0, 20));
    madd(reg_tmp, reg_a0, reg_b5, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 20));
    ldr(reg_tmp, ptr(reg_c1, 20));
    madd(reg_tmp, reg_a1, reg_b5, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 20));

    ldr(reg_tmp, ptr(reg_c0, 24));
    madd(reg_tmp, reg_a0, reg_b6, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 24));
    ldr(reg_tmp, ptr(reg_c1, 24));
    madd(reg_tmp, reg_a1, reg_b6, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 24));

    ldr(reg_tmp, ptr(reg_c0, 28));
    madd(reg_tmp, reg_a0, reg_b7, reg_tmp);
    str(reg_tmp, ptr(reg_c0, 28));
    ldr(reg_tmp, ptr(reg_c1, 28));
    madd(reg_tmp, reg_a1, reg_b7, reg_tmp);
    str(reg_tmp, ptr(reg_c1, 28));

    add(reg_src0, reg_src0, 1);
    add(reg_src1, reg_src1, 1);
    add(reg_wei, reg_wei, 8);

    subs(reg_k, reg_k, 1);
    b(NE, tail_loop);

    L(done);
    postamble();
}

jit_int8_brgemm_kernel_2x8_dot_packed_strided::jit_int8_brgemm_kernel_2x8_dot_packed_strided() : jit_generator() {}

void jit_int8_brgemm_kernel_2x8_dot_packed_strided::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_2x8_dot_packed_strided::generate() {
    preamble();

    const XReg reg_src = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_src_stride = abi_param5;
    const XReg reg_ldc = abi_param6;
    const XReg reg_accum = abi_param7;

    const XReg reg_src0 = x10;
    const XReg reg_src1 = x11;

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

    mov(reg_src0, reg_src);
    add(reg_src1, reg_src0, reg_src_stride);

    mov(reg_c0, reg_dst);
    add(reg_c1, reg_c0, reg_ldc);

    Label loop64;
    Label check32;
    Label loop32;
    Label loop16;
    Label reduce_store;
    Label done;

    cmp(reg_k, 64);
    b(LT, check32);

    L(loop64);
    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(VReg16BList(v_w0, v_w3), post_ptr(reg_wei, 64));
    ld1(VReg16BList(v_w4, v_w7), post_ptr(reg_wei, 64));
    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
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
    ld1(VReg16BList(v_w0, v_w3), post_ptr(reg_wei, 64));
    ld1(VReg16BList(v_w4, v_w7), post_ptr(reg_wei, 64));
    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
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
    ld1(VReg16BList(v_w0, v_w3), post_ptr(reg_wei, 64));
    ld1(VReg16BList(v_w4, v_w7), post_ptr(reg_wei, 64));
    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
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
    ld1(VReg16BList(v_w0, v_w3), post_ptr(reg_wei, 64));
    ld1(VReg16BList(v_w4, v_w7), post_ptr(reg_wei, 64));
    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
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

    subs(reg_k, reg_k, 64);
    cmp(reg_k, 64);
    b(GE, loop64);

    L(check32);
    cmp(reg_k, 32);
    b(LT, loop16);

    L(loop32);
    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(VReg16BList(v_w0, v_w3), post_ptr(reg_wei, 64));
    ld1(VReg16BList(v_w4, v_w7), post_ptr(reg_wei, 64));
    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
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
    ld1(VReg16BList(v_w0, v_w3), post_ptr(reg_wei, 64));
    ld1(VReg16BList(v_w4, v_w7), post_ptr(reg_wei, 64));
    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
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

    subs(reg_k, reg_k, 32);
    b(GE, loop32);

    L(loop16);
    cmp(reg_k, 16);
    b(LT, reduce_store);
    ld1(v_src0, ptr(reg_src0));
    ld1(v_src1, ptr(reg_src1));
    ld1(VReg16BList(v_w0, v_w3), post_ptr(reg_wei, 64));
    ld1(VReg16BList(v_w4, v_w7), post_ptr(reg_wei, 64));
    add(reg_src0, reg_src0, 16);
    add(reg_src1, reg_src1, 16);
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
    Label store_no_acc;
    Label store_acc;
    cbnz(reg_accum, store_acc);

    L(store_no_acc);
    const VReg v_tmp(0);
    const SReg s_tmp(v_tmp.getIdx());
    const WReg reg_tmp = w9;
    const WReg reg_c_acc = w0;
    const WReg reg_accum_w(reg_accum.getIdx());

    addv(s_tmp, v_acc00);
    umov(reg_tmp, v_tmp.s[0]);
    str(reg_tmp, ptr(reg_c0));
    addv(s_tmp, v_acc01);
    umov(reg_tmp, v_tmp.s[0]);
    str(reg_tmp, ptr(reg_c0, 4));
    addv(s_tmp, v_acc02);
    umov(reg_tmp, v_tmp.s[0]);
    str(reg_tmp, ptr(reg_c0, 8));
    addv(s_tmp, v_acc03);
    umov(reg_tmp, v_tmp.s[0]);
    str(reg_tmp, ptr(reg_c0, 12));
    addv(s_tmp, v_acc04);
    umov(reg_tmp, v_tmp.s[0]);
    str(reg_tmp, ptr(reg_c0, 16));
    addv(s_tmp, v_acc05);
    umov(reg_tmp, v_tmp.s[0]);
    str(reg_tmp, ptr(reg_c0, 20));
    addv(s_tmp, v_acc06);
    umov(reg_tmp, v_tmp.s[0]);
    str(reg_tmp, ptr(reg_c0, 24));
    addv(s_tmp, v_acc07);
    umov(reg_tmp, v_tmp.s[0]);
    str(reg_tmp, ptr(reg_c0, 28));

    addv(s_tmp, v_acc10);
    umov(reg_tmp, v_tmp.s[0]);
    str(reg_tmp, ptr(reg_c1));
    addv(s_tmp, v_acc11);
    umov(reg_tmp, v_tmp.s[0]);
    str(reg_tmp, ptr(reg_c1, 4));
    addv(s_tmp, v_acc12);
    umov(reg_tmp, v_tmp.s[0]);
    str(reg_tmp, ptr(reg_c1, 8));
    addv(s_tmp, v_acc13);
    umov(reg_tmp, v_tmp.s[0]);
    str(reg_tmp, ptr(reg_c1, 12));
    addv(s_tmp, v_acc14);
    umov(reg_tmp, v_tmp.s[0]);
    str(reg_tmp, ptr(reg_c1, 16));
    addv(s_tmp, v_acc15);
    umov(reg_tmp, v_tmp.s[0]);
    str(reg_tmp, ptr(reg_c1, 20));
    addv(s_tmp, v_acc16);
    umov(reg_tmp, v_tmp.s[0]);
    str(reg_tmp, ptr(reg_c1, 24));
    addv(s_tmp, v_acc17);
    umov(reg_tmp, v_tmp.s[0]);
    str(reg_tmp, ptr(reg_c1, 28));
    b(done);

    L(store_acc);
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

    L(done);
    postamble();
}

jit_int8_brgemm_kernel_2x8_dot_packed_strided_interleaved4::jit_int8_brgemm_kernel_2x8_dot_packed_strided_interleaved4()
    : jit_generator() {}

void jit_int8_brgemm_kernel_2x8_dot_packed_strided_interleaved4::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_2x8_dot_packed_strided_interleaved4::generate() {
    preamble();

    const XReg reg_src = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_src_stride = abi_param5;
    const XReg reg_ldc = abi_param6;
    const XReg reg_accum = abi_param7;

    const XReg reg_src0 = x10;
    const XReg reg_src1 = x11;
    const XReg reg_c0 = x12;
    const XReg reg_c1 = x13;
    const XReg reg_kb = x14;

    const WReg reg_tmp0 = w15;
    const WReg reg_tmp1 = w16;

    const VReg4S v_src0s = VReg4S(0);
    const VReg4S v_src1s = VReg4S(1);
    const VReg16B v_src0b(v_src0s.getIdx());
    const VReg16B v_src1b(v_src1s.getIdx());
    const VReg16B v_w0 = VReg16B(2);
    const VReg16B v_w1 = VReg16B(3);

    const VReg4S v_acc00 = VReg4S(16);
    const VReg4S v_acc01 = VReg4S(17);
    const VReg4S v_acc10 = VReg4S(18);
    const VReg4S v_acc11 = VReg4S(19);
    const VReg16B v_acc00b(v_acc00.getIdx());
    const VReg16B v_acc01b(v_acc01.getIdx());
    const VReg16B v_acc10b(v_acc10.getIdx());
    const VReg16B v_acc11b(v_acc11.getIdx());

    const VReg4S v_tmp0 = VReg4S(20);
    const VReg4S v_tmp1 = VReg4S(21);
    const QReg q_acc00(v_acc00.getIdx());
    const QReg q_acc01(v_acc01.getIdx());
    const QReg q_acc10(v_acc10.getIdx());
    const QReg q_acc11(v_acc11.getIdx());
    const QReg q_tmp0(v_tmp0.getIdx());
    const QReg q_tmp1(v_tmp1.getIdx());

    auto prefetch_next = [&]() {
        prfm(PLDL1KEEP, ptr(reg_wei, 256));
        prfm(PLDL1KEEP, ptr(reg_src0, 64));
        prfm(PLDL1KEEP, ptr(reg_src1, 64));
    };

    eor(v_acc00b, v_acc00b, v_acc00b);
    eor(v_acc01b, v_acc01b, v_acc01b);
    eor(v_acc10b, v_acc10b, v_acc10b);
    eor(v_acc11b, v_acc11b, v_acc11b);

    mov(reg_src0, reg_src);
    add(reg_src1, reg_src0, reg_src_stride);

    mov(reg_c0, reg_dst);
    add(reg_c1, reg_c0, reg_ldc);

    lsr(reg_kb, reg_k, 2);

    Label loop;
    Label store_acc;
    Label done;

    cbz(reg_kb, done);

    L(loop);
    prefetch_next();
    ldr(reg_tmp0, ptr(reg_src0));
    ldr(reg_tmp1, ptr(reg_src1));
    add(reg_src0, reg_src0, 4);
    add(reg_src1, reg_src1, 4);
    dup(v_src0s, reg_tmp0);
    dup(v_src1s, reg_tmp1);
    ld1(v_w0, post_ptr(reg_wei, 16));
    ld1(v_w1, post_ptr(reg_wei, 16));
    sdot(v_acc00, v_src0b, v_w0);
    sdot(v_acc01, v_src0b, v_w1);
    sdot(v_acc10, v_src1b, v_w0);
    sdot(v_acc11, v_src1b, v_w1);
    subs(reg_kb, reg_kb, 1);
    b(NE, loop);

    cbnz(reg_accum, store_acc);
    str(q_acc00, ptr(reg_c0));
    str(q_acc01, ptr(reg_c0, 16));
    str(q_acc10, ptr(reg_c1));
    str(q_acc11, ptr(reg_c1, 16));
    b(done);

    L(store_acc);
    ldr(q_tmp0, ptr(reg_c0));
    add(v_acc00, v_acc00, v_tmp0);
    str(q_acc00, ptr(reg_c0));
    ldr(q_tmp1, ptr(reg_c0, 16));
    add(v_acc01, v_acc01, v_tmp1);
    str(q_acc01, ptr(reg_c0, 16));

    ldr(q_tmp0, ptr(reg_c1));
    add(v_acc10, v_acc10, v_tmp0);
    str(q_acc10, ptr(reg_c1));
    ldr(q_tmp1, ptr(reg_c1, 16));
    add(v_acc11, v_acc11, v_tmp1);
    str(q_acc11, ptr(reg_c1, 16));

    L(done);
    postamble();
}

jit_int8_brgemm_kernel_2x16_dot_packed_strided_interleaved4::jit_int8_brgemm_kernel_2x16_dot_packed_strided_interleaved4()
    : jit_generator() {}

void jit_int8_brgemm_kernel_2x16_dot_packed_strided_interleaved4::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_2x16_dot_packed_strided_interleaved4::generate() {
    preamble();

    const XReg reg_src = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_src_stride = abi_param5;
    const XReg reg_ldc = abi_param6;
    const XReg reg_accum = abi_param7;

    const XReg reg_src0 = x10;
    const XReg reg_src1 = x11;
    const XReg reg_c0 = x12;
    const XReg reg_c1 = x13;
    const XReg reg_kb = x14;

    const WReg reg_tmp0 = w15;
    const WReg reg_tmp1 = w16;

    const VReg4S v_src0s = VReg4S(0);
    const VReg4S v_src1s = VReg4S(1);
    const VReg16B v_src0b(v_src0s.getIdx());
    const VReg16B v_src1b(v_src1s.getIdx());
    const VReg16B v_w0 = VReg16B(2);
    const VReg16B v_w1 = VReg16B(3);
    const VReg16B v_w2 = VReg16B(4);
    const VReg16B v_w3 = VReg16B(5);

    const VReg4S v_acc00 = VReg4S(16);
    const VReg4S v_acc01 = VReg4S(17);
    const VReg4S v_acc02 = VReg4S(18);
    const VReg4S v_acc03 = VReg4S(19);
    const VReg4S v_acc10 = VReg4S(20);
    const VReg4S v_acc11 = VReg4S(21);
    const VReg4S v_acc12 = VReg4S(22);
    const VReg4S v_acc13 = VReg4S(23);
    const VReg16B v_acc00b(v_acc00.getIdx());
    const VReg16B v_acc01b(v_acc01.getIdx());
    const VReg16B v_acc02b(v_acc02.getIdx());
    const VReg16B v_acc03b(v_acc03.getIdx());
    const VReg16B v_acc10b(v_acc10.getIdx());
    const VReg16B v_acc11b(v_acc11.getIdx());
    const VReg16B v_acc12b(v_acc12.getIdx());
    const VReg16B v_acc13b(v_acc13.getIdx());

    const VReg4S v_tmp0 = VReg4S(24);
    const VReg4S v_tmp1 = VReg4S(25);
    const QReg q_acc00(v_acc00.getIdx());
    const QReg q_acc01(v_acc01.getIdx());
    const QReg q_acc02(v_acc02.getIdx());
    const QReg q_acc03(v_acc03.getIdx());
    const QReg q_acc10(v_acc10.getIdx());
    const QReg q_acc11(v_acc11.getIdx());
    const QReg q_acc12(v_acc12.getIdx());
    const QReg q_acc13(v_acc13.getIdx());
    const QReg q_tmp0(v_tmp0.getIdx());
    const QReg q_tmp1(v_tmp1.getIdx());

    auto prefetch_next = [&]() {
        prfm(PLDL1KEEP, ptr(reg_wei, 256));
        prfm(PLDL1KEEP, ptr(reg_src0, 64));
        prfm(PLDL1KEEP, ptr(reg_src1, 64));
    };

    eor(v_acc00b, v_acc00b, v_acc00b);
    eor(v_acc01b, v_acc01b, v_acc01b);
    eor(v_acc02b, v_acc02b, v_acc02b);
    eor(v_acc03b, v_acc03b, v_acc03b);
    eor(v_acc10b, v_acc10b, v_acc10b);
    eor(v_acc11b, v_acc11b, v_acc11b);
    eor(v_acc12b, v_acc12b, v_acc12b);
    eor(v_acc13b, v_acc13b, v_acc13b);

    mov(reg_src0, reg_src);
    add(reg_src1, reg_src0, reg_src_stride);

    mov(reg_c0, reg_dst);
    add(reg_c1, reg_c0, reg_ldc);

    lsr(reg_kb, reg_k, 2);

    Label loop;
    Label store_acc;
    Label done;

    cbz(reg_kb, done);

    L(loop);
    prefetch_next();
    ldr(reg_tmp0, ptr(reg_src0));
    ldr(reg_tmp1, ptr(reg_src1));
    add(reg_src0, reg_src0, 4);
    add(reg_src1, reg_src1, 4);
    dup(v_src0s, reg_tmp0);
    dup(v_src1s, reg_tmp1);
    ld1(v_w0, post_ptr(reg_wei, 16));
    ld1(v_w1, post_ptr(reg_wei, 16));
    ld1(v_w2, post_ptr(reg_wei, 16));
    ld1(v_w3, post_ptr(reg_wei, 16));
    sdot(v_acc00, v_src0b, v_w0);
    sdot(v_acc01, v_src0b, v_w1);
    sdot(v_acc02, v_src0b, v_w2);
    sdot(v_acc03, v_src0b, v_w3);
    sdot(v_acc10, v_src1b, v_w0);
    sdot(v_acc11, v_src1b, v_w1);
    sdot(v_acc12, v_src1b, v_w2);
    sdot(v_acc13, v_src1b, v_w3);
    subs(reg_kb, reg_kb, 1);
    b(NE, loop);

    cbnz(reg_accum, store_acc);
    str(q_acc00, ptr(reg_c0));
    str(q_acc01, ptr(reg_c0, 16));
    str(q_acc02, ptr(reg_c0, 32));
    str(q_acc03, ptr(reg_c0, 48));
    str(q_acc10, ptr(reg_c1));
    str(q_acc11, ptr(reg_c1, 16));
    str(q_acc12, ptr(reg_c1, 32));
    str(q_acc13, ptr(reg_c1, 48));
    b(done);

    L(store_acc);
    ldr(q_tmp0, ptr(reg_c0));
    add(v_acc00, v_acc00, v_tmp0);
    str(q_acc00, ptr(reg_c0));
    ldr(q_tmp1, ptr(reg_c0, 16));
    add(v_acc01, v_acc01, v_tmp1);
    str(q_acc01, ptr(reg_c0, 16));
    ldr(q_tmp0, ptr(reg_c0, 32));
    add(v_acc02, v_acc02, v_tmp0);
    str(q_acc02, ptr(reg_c0, 32));
    ldr(q_tmp1, ptr(reg_c0, 48));
    add(v_acc03, v_acc03, v_tmp1);
    str(q_acc03, ptr(reg_c0, 48));

    ldr(q_tmp0, ptr(reg_c1));
    add(v_acc10, v_acc10, v_tmp0);
    str(q_acc10, ptr(reg_c1));
    ldr(q_tmp1, ptr(reg_c1, 16));
    add(v_acc11, v_acc11, v_tmp1);
    str(q_acc11, ptr(reg_c1, 16));
    ldr(q_tmp0, ptr(reg_c1, 32));
    add(v_acc12, v_acc12, v_tmp0);
    str(q_acc12, ptr(reg_c1, 32));
    ldr(q_tmp1, ptr(reg_c1, 48));
    add(v_acc13, v_acc13, v_tmp1);
    str(q_acc13, ptr(reg_c1, 48));

    L(done);
    postamble();
}

jit_int8_brgemm_kernel_2x32_dot_packed_strided_interleaved4::jit_int8_brgemm_kernel_2x32_dot_packed_strided_interleaved4()
    : jit_generator() {}

void jit_int8_brgemm_kernel_2x32_dot_packed_strided_interleaved4::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_2x32_dot_packed_strided_interleaved4::generate() {
    preamble();

    const XReg reg_src = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_src_stride = abi_param5;
    const XReg reg_ldc = abi_param6;
    const XReg reg_accum = abi_param7;

    const XReg reg_src0 = x10;
    const XReg reg_src1 = x11;
    const XReg reg_c0 = x12;
    const XReg reg_c1 = x13;
    const XReg reg_kb = x14;
    const XReg reg_kb_tail = x15;

    const VReg4S v_src0s = VReg4S(0);
    const VReg4S v_src1s = VReg4S(1);
    const VReg16B v_src0b(v_src0s.getIdx());
    const VReg16B v_src1b(v_src1s.getIdx());
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

    const QReg q_acc00(v_acc00.getIdx());
    const QReg q_acc01(v_acc01.getIdx());
    const QReg q_acc02(v_acc02.getIdx());
    const QReg q_acc03(v_acc03.getIdx());
    const QReg q_acc04(v_acc04.getIdx());
    const QReg q_acc05(v_acc05.getIdx());
    const QReg q_acc06(v_acc06.getIdx());
    const QReg q_acc07(v_acc07.getIdx());
    const QReg q_acc10(v_acc10.getIdx());
    const QReg q_acc11(v_acc11.getIdx());
    const QReg q_acc12(v_acc12.getIdx());
    const QReg q_acc13(v_acc13.getIdx());
    const QReg q_acc14(v_acc14.getIdx());
    const QReg q_acc15(v_acc15.getIdx());
    const QReg q_acc16(v_acc16.getIdx());
    const QReg q_acc17(v_acc17.getIdx());
    const QReg q_tmp0(v_src0s.getIdx());
    const QReg q_tmp1(v_src1s.getIdx());

    auto prefetch_next = [&]() {
        prfm(PLDL1KEEP, ptr(reg_wei, 256));
        prfm(PLDL1KEEP, ptr(reg_src0, 64));
        prfm(PLDL1KEEP, ptr(reg_src1, 64));
    };

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

    mov(reg_src0, reg_src);
    add(reg_src1, reg_src0, reg_src_stride);

    mov(reg_c0, reg_dst);
    add(reg_c1, reg_c0, reg_ldc);

    lsr(reg_kb, reg_k, 3);
    and_(reg_kb_tail, reg_k, 4);

    Label loop;
    Label tail;
    Label store_acc;
    Label done;

    cbz(reg_kb, tail);

    L(loop);
    prefetch_next();
    ld1r(v_src0s, post_ptr(reg_src0, 4));
    ld1r(v_src1s, post_ptr(reg_src1, 4));
    ld1(v_w0, post_ptr(reg_wei, 16));
    ld1(v_w1, post_ptr(reg_wei, 16));
    ld1(v_w2, post_ptr(reg_wei, 16));
    ld1(v_w3, post_ptr(reg_wei, 16));
    ld1(v_w4, post_ptr(reg_wei, 16));
    ld1(v_w5, post_ptr(reg_wei, 16));
    ld1(v_w6, post_ptr(reg_wei, 16));
    ld1(v_w7, post_ptr(reg_wei, 16));
    sdot(v_acc00, v_src0b, v_w0);
    sdot(v_acc01, v_src0b, v_w1);
    sdot(v_acc02, v_src0b, v_w2);
    sdot(v_acc03, v_src0b, v_w3);
    sdot(v_acc04, v_src0b, v_w4);
    sdot(v_acc05, v_src0b, v_w5);
    sdot(v_acc06, v_src0b, v_w6);
    sdot(v_acc07, v_src0b, v_w7);
    sdot(v_acc10, v_src1b, v_w0);
    sdot(v_acc11, v_src1b, v_w1);
    sdot(v_acc12, v_src1b, v_w2);
    sdot(v_acc13, v_src1b, v_w3);
    sdot(v_acc14, v_src1b, v_w4);
    sdot(v_acc15, v_src1b, v_w5);
    sdot(v_acc16, v_src1b, v_w6);
    sdot(v_acc17, v_src1b, v_w7);
    ld1r(v_src0s, post_ptr(reg_src0, 4));
    ld1r(v_src1s, post_ptr(reg_src1, 4));
    ld1(v_w0, post_ptr(reg_wei, 16));
    ld1(v_w1, post_ptr(reg_wei, 16));
    ld1(v_w2, post_ptr(reg_wei, 16));
    ld1(v_w3, post_ptr(reg_wei, 16));
    ld1(v_w4, post_ptr(reg_wei, 16));
    ld1(v_w5, post_ptr(reg_wei, 16));
    ld1(v_w6, post_ptr(reg_wei, 16));
    ld1(v_w7, post_ptr(reg_wei, 16));
    sdot(v_acc00, v_src0b, v_w0);
    sdot(v_acc01, v_src0b, v_w1);
    sdot(v_acc02, v_src0b, v_w2);
    sdot(v_acc03, v_src0b, v_w3);
    sdot(v_acc04, v_src0b, v_w4);
    sdot(v_acc05, v_src0b, v_w5);
    sdot(v_acc06, v_src0b, v_w6);
    sdot(v_acc07, v_src0b, v_w7);
    sdot(v_acc10, v_src1b, v_w0);
    sdot(v_acc11, v_src1b, v_w1);
    sdot(v_acc12, v_src1b, v_w2);
    sdot(v_acc13, v_src1b, v_w3);
    sdot(v_acc14, v_src1b, v_w4);
    sdot(v_acc15, v_src1b, v_w5);
    sdot(v_acc16, v_src1b, v_w6);
    sdot(v_acc17, v_src1b, v_w7);
    subs(reg_kb, reg_kb, 1);
    b(NE, loop);

    L(tail);
    cbz(reg_kb_tail, store_acc);
    prefetch_next();
    ld1r(v_src0s, post_ptr(reg_src0, 4));
    ld1r(v_src1s, post_ptr(reg_src1, 4));
    ld1(v_w0, post_ptr(reg_wei, 16));
    ld1(v_w1, post_ptr(reg_wei, 16));
    ld1(v_w2, post_ptr(reg_wei, 16));
    ld1(v_w3, post_ptr(reg_wei, 16));
    ld1(v_w4, post_ptr(reg_wei, 16));
    ld1(v_w5, post_ptr(reg_wei, 16));
    ld1(v_w6, post_ptr(reg_wei, 16));
    ld1(v_w7, post_ptr(reg_wei, 16));
    sdot(v_acc00, v_src0b, v_w0);
    sdot(v_acc01, v_src0b, v_w1);
    sdot(v_acc02, v_src0b, v_w2);
    sdot(v_acc03, v_src0b, v_w3);
    sdot(v_acc04, v_src0b, v_w4);
    sdot(v_acc05, v_src0b, v_w5);
    sdot(v_acc06, v_src0b, v_w6);
    sdot(v_acc07, v_src0b, v_w7);
    sdot(v_acc10, v_src1b, v_w0);
    sdot(v_acc11, v_src1b, v_w1);
    sdot(v_acc12, v_src1b, v_w2);
    sdot(v_acc13, v_src1b, v_w3);
    sdot(v_acc14, v_src1b, v_w4);
    sdot(v_acc15, v_src1b, v_w5);
    sdot(v_acc16, v_src1b, v_w6);
    sdot(v_acc17, v_src1b, v_w7);

    cbnz(reg_accum, store_acc);
    str(q_acc00, ptr(reg_c0));
    str(q_acc01, ptr(reg_c0, 16));
    str(q_acc02, ptr(reg_c0, 32));
    str(q_acc03, ptr(reg_c0, 48));
    str(q_acc04, ptr(reg_c0, 64));
    str(q_acc05, ptr(reg_c0, 80));
    str(q_acc06, ptr(reg_c0, 96));
    str(q_acc07, ptr(reg_c0, 112));
    str(q_acc10, ptr(reg_c1));
    str(q_acc11, ptr(reg_c1, 16));
    str(q_acc12, ptr(reg_c1, 32));
    str(q_acc13, ptr(reg_c1, 48));
    str(q_acc14, ptr(reg_c1, 64));
    str(q_acc15, ptr(reg_c1, 80));
    str(q_acc16, ptr(reg_c1, 96));
    str(q_acc17, ptr(reg_c1, 112));
    b(done);

    L(store_acc);
    ldr(q_tmp0, ptr(reg_c0));
    add(v_acc00, v_acc00, v_src0s);
    str(q_acc00, ptr(reg_c0));
    ldr(q_tmp1, ptr(reg_c0, 16));
    add(v_acc01, v_acc01, v_src1s);
    str(q_acc01, ptr(reg_c0, 16));
    ldr(q_tmp0, ptr(reg_c0, 32));
    add(v_acc02, v_acc02, v_src0s);
    str(q_acc02, ptr(reg_c0, 32));
    ldr(q_tmp1, ptr(reg_c0, 48));
    add(v_acc03, v_acc03, v_src1s);
    str(q_acc03, ptr(reg_c0, 48));
    ldr(q_tmp0, ptr(reg_c0, 64));
    add(v_acc04, v_acc04, v_src0s);
    str(q_acc04, ptr(reg_c0, 64));
    ldr(q_tmp1, ptr(reg_c0, 80));
    add(v_acc05, v_acc05, v_src1s);
    str(q_acc05, ptr(reg_c0, 80));
    ldr(q_tmp0, ptr(reg_c0, 96));
    add(v_acc06, v_acc06, v_src0s);
    str(q_acc06, ptr(reg_c0, 96));
    ldr(q_tmp1, ptr(reg_c0, 112));
    add(v_acc07, v_acc07, v_src1s);
    str(q_acc07, ptr(reg_c0, 112));

    ldr(q_tmp0, ptr(reg_c1));
    add(v_acc10, v_acc10, v_src0s);
    str(q_acc10, ptr(reg_c1));
    ldr(q_tmp1, ptr(reg_c1, 16));
    add(v_acc11, v_acc11, v_src1s);
    str(q_acc11, ptr(reg_c1, 16));
    ldr(q_tmp0, ptr(reg_c1, 32));
    add(v_acc12, v_acc12, v_src0s);
    str(q_acc12, ptr(reg_c1, 32));
    ldr(q_tmp1, ptr(reg_c1, 48));
    add(v_acc13, v_acc13, v_src1s);
    str(q_acc13, ptr(reg_c1, 48));
    ldr(q_tmp0, ptr(reg_c1, 64));
    add(v_acc14, v_acc14, v_src0s);
    str(q_acc14, ptr(reg_c1, 64));
    ldr(q_tmp1, ptr(reg_c1, 80));
    add(v_acc15, v_acc15, v_src1s);
    str(q_acc15, ptr(reg_c1, 80));
    ldr(q_tmp0, ptr(reg_c1, 96));
    add(v_acc16, v_acc16, v_src0s);
    str(q_acc16, ptr(reg_c1, 96));
    ldr(q_tmp1, ptr(reg_c1, 112));
    add(v_acc17, v_acc17, v_src1s);
    str(q_acc17, ptr(reg_c1, 112));

    L(done);
    postamble();
}

jit_int8_brgemm_kernel_4x16_dot_packed_strided_interleaved4::jit_int8_brgemm_kernel_4x16_dot_packed_strided_interleaved4()
    : jit_generator() {}

void jit_int8_brgemm_kernel_4x16_dot_packed_strided_interleaved4::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_4x16_dot_packed_strided_interleaved4::generate() {
    preamble();

    const XReg reg_src = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_src_stride = abi_param5;
    const XReg reg_ldc = abi_param6;
    const XReg reg_accum = abi_param7;

    const XReg reg_src0 = x10;
    const XReg reg_src1 = x11;
    const XReg reg_src2 = x12;
    const XReg reg_src3 = x13;
    const XReg reg_c0 = x14;
    const XReg reg_c1 = x15;
    const XReg reg_c2 = x16;
    const XReg reg_c3 = x17;
    const XReg reg_kb = x18;
    const XReg reg_kb4 = x23;
    const XReg reg_kb_tail = x24;

    const VReg4S v_src0s = VReg4S(0);
    const VReg4S v_src1s = VReg4S(1);
    const VReg4S v_src2s = VReg4S(2);
    const VReg4S v_src3s = VReg4S(3);
    const VReg16B v_src0b(v_src0s.getIdx());
    const VReg16B v_src1b(v_src1s.getIdx());
    const VReg16B v_src2b(v_src2s.getIdx());
    const VReg16B v_src3b(v_src3s.getIdx());
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

    const VReg4S v_tmp0 = VReg4S(8);
    const VReg4S v_tmp1 = VReg4S(9);
    const QReg q_acc00(v_acc00.getIdx());
    const QReg q_acc01(v_acc01.getIdx());
    const QReg q_acc02(v_acc02.getIdx());
    const QReg q_acc03(v_acc03.getIdx());
    const QReg q_acc10(v_acc10.getIdx());
    const QReg q_acc11(v_acc11.getIdx());
    const QReg q_acc12(v_acc12.getIdx());
    const QReg q_acc13(v_acc13.getIdx());
    const QReg q_acc20(v_acc20.getIdx());
    const QReg q_acc21(v_acc21.getIdx());
    const QReg q_acc22(v_acc22.getIdx());
    const QReg q_acc23(v_acc23.getIdx());
    const QReg q_acc30(v_acc30.getIdx());
    const QReg q_acc31(v_acc31.getIdx());
    const QReg q_acc32(v_acc32.getIdx());
    const QReg q_acc33(v_acc33.getIdx());
    const QReg q_tmp0(v_tmp0.getIdx());
    const QReg q_tmp1(v_tmp1.getIdx());

    auto prefetch_next = [&]() {
        prfm(PLDL1KEEP, ptr(reg_wei, 256));
        prfm(PLDL1KEEP, ptr(reg_src0, 64));
        prfm(PLDL1KEEP, ptr(reg_src1, 64));
        prfm(PLDL1KEEP, ptr(reg_src2, 64));
        prfm(PLDL1KEEP, ptr(reg_src3, 64));
    };

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

    mov(reg_src0, reg_src);
    add(reg_src1, reg_src0, reg_src_stride);
    add(reg_src2, reg_src1, reg_src_stride);
    add(reg_src3, reg_src2, reg_src_stride);

    mov(reg_c0, reg_dst);
    add(reg_c1, reg_c0, reg_ldc);
    add(reg_c2, reg_c1, reg_ldc);
    add(reg_c3, reg_c2, reg_ldc);

    lsr(reg_kb, reg_k, 2);

    Label loop4;
    Label loop1;
    Label store_acc;
    Label done;

    cbz(reg_kb, done);

    lsr(reg_kb4, reg_kb, 2);
    lsl(reg_kb_tail, reg_kb4, 2);
    sub(reg_kb, reg_kb, reg_kb_tail);

    cbz(reg_kb4, loop1);

    L(loop4);
    prefetch_next();
    for (int unroll = 0; unroll < 4; ++unroll) {
        ld1r(v_src0s, post_ptr(reg_src0, 4));
        ld1r(v_src1s, post_ptr(reg_src1, 4));
        ld1r(v_src2s, post_ptr(reg_src2, 4));
        ld1r(v_src3s, post_ptr(reg_src3, 4));
        ld1(VReg16BList(v_w0, v_w1), post_ptr(reg_wei, 32));
        ld1(VReg16BList(v_w2, v_w3), post_ptr(reg_wei, 32));
        sdot(v_acc00, v_src0b, v_w0);
        sdot(v_acc01, v_src0b, v_w1);
        sdot(v_acc02, v_src0b, v_w2);
        sdot(v_acc03, v_src0b, v_w3);
        sdot(v_acc10, v_src1b, v_w0);
        sdot(v_acc11, v_src1b, v_w1);
        sdot(v_acc12, v_src1b, v_w2);
        sdot(v_acc13, v_src1b, v_w3);
        sdot(v_acc20, v_src2b, v_w0);
        sdot(v_acc21, v_src2b, v_w1);
        sdot(v_acc22, v_src2b, v_w2);
        sdot(v_acc23, v_src2b, v_w3);
        sdot(v_acc30, v_src3b, v_w0);
        sdot(v_acc31, v_src3b, v_w1);
        sdot(v_acc32, v_src3b, v_w2);
        sdot(v_acc33, v_src3b, v_w3);
    }
    subs(reg_kb4, reg_kb4, 1);
    b(NE, loop4);

    L(loop1);
    cbz(reg_kb, done);
    prefetch_next();
    ld1r(v_src0s, post_ptr(reg_src0, 4));
    ld1r(v_src1s, post_ptr(reg_src1, 4));
    ld1r(v_src2s, post_ptr(reg_src2, 4));
    ld1r(v_src3s, post_ptr(reg_src3, 4));
    ld1(VReg16BList(v_w0, v_w1), post_ptr(reg_wei, 32));
    ld1(VReg16BList(v_w2, v_w3), post_ptr(reg_wei, 32));
    sdot(v_acc00, v_src0b, v_w0);
    sdot(v_acc01, v_src0b, v_w1);
    sdot(v_acc02, v_src0b, v_w2);
    sdot(v_acc03, v_src0b, v_w3);
    sdot(v_acc10, v_src1b, v_w0);
    sdot(v_acc11, v_src1b, v_w1);
    sdot(v_acc12, v_src1b, v_w2);
    sdot(v_acc13, v_src1b, v_w3);
    sdot(v_acc20, v_src2b, v_w0);
    sdot(v_acc21, v_src2b, v_w1);
    sdot(v_acc22, v_src2b, v_w2);
    sdot(v_acc23, v_src2b, v_w3);
    sdot(v_acc30, v_src3b, v_w0);
    sdot(v_acc31, v_src3b, v_w1);
    sdot(v_acc32, v_src3b, v_w2);
    sdot(v_acc33, v_src3b, v_w3);
    subs(reg_kb, reg_kb, 1);
    b(NE, loop1);

    cbnz(reg_accum, store_acc);
    str(q_acc00, ptr(reg_c0));
    str(q_acc01, ptr(reg_c0, 16));
    str(q_acc02, ptr(reg_c0, 32));
    str(q_acc03, ptr(reg_c0, 48));
    str(q_acc10, ptr(reg_c1));
    str(q_acc11, ptr(reg_c1, 16));
    str(q_acc12, ptr(reg_c1, 32));
    str(q_acc13, ptr(reg_c1, 48));
    str(q_acc20, ptr(reg_c2));
    str(q_acc21, ptr(reg_c2, 16));
    str(q_acc22, ptr(reg_c2, 32));
    str(q_acc23, ptr(reg_c2, 48));
    str(q_acc30, ptr(reg_c3));
    str(q_acc31, ptr(reg_c3, 16));
    str(q_acc32, ptr(reg_c3, 32));
    str(q_acc33, ptr(reg_c3, 48));
    b(done);

    L(store_acc);
    ldr(q_tmp0, ptr(reg_c0));
    add(v_acc00, v_acc00, v_tmp0);
    str(q_acc00, ptr(reg_c0));
    ldr(q_tmp1, ptr(reg_c0, 16));
    add(v_acc01, v_acc01, v_tmp1);
    str(q_acc01, ptr(reg_c0, 16));
    ldr(q_tmp0, ptr(reg_c0, 32));
    add(v_acc02, v_acc02, v_tmp0);
    str(q_acc02, ptr(reg_c0, 32));
    ldr(q_tmp1, ptr(reg_c0, 48));
    add(v_acc03, v_acc03, v_tmp1);
    str(q_acc03, ptr(reg_c0, 48));

    ldr(q_tmp0, ptr(reg_c1));
    add(v_acc10, v_acc10, v_tmp0);
    str(q_acc10, ptr(reg_c1));
    ldr(q_tmp1, ptr(reg_c1, 16));
    add(v_acc11, v_acc11, v_tmp1);
    str(q_acc11, ptr(reg_c1, 16));
    ldr(q_tmp0, ptr(reg_c1, 32));
    add(v_acc12, v_acc12, v_tmp0);
    str(q_acc12, ptr(reg_c1, 32));
    ldr(q_tmp1, ptr(reg_c1, 48));
    add(v_acc13, v_acc13, v_tmp1);
    str(q_acc13, ptr(reg_c1, 48));

    ldr(q_tmp0, ptr(reg_c2));
    add(v_acc20, v_acc20, v_tmp0);
    str(q_acc20, ptr(reg_c2));
    ldr(q_tmp1, ptr(reg_c2, 16));
    add(v_acc21, v_acc21, v_tmp1);
    str(q_acc21, ptr(reg_c2, 16));
    ldr(q_tmp0, ptr(reg_c2, 32));
    add(v_acc22, v_acc22, v_tmp0);
    str(q_acc22, ptr(reg_c2, 32));
    ldr(q_tmp1, ptr(reg_c2, 48));
    add(v_acc23, v_acc23, v_tmp1);
    str(q_acc23, ptr(reg_c2, 48));

    ldr(q_tmp0, ptr(reg_c3));
    add(v_acc30, v_acc30, v_tmp0);
    str(q_acc30, ptr(reg_c3));
    ldr(q_tmp1, ptr(reg_c3, 16));
    add(v_acc31, v_acc31, v_tmp1);
    str(q_acc31, ptr(reg_c3, 16));
    ldr(q_tmp0, ptr(reg_c3, 32));
    add(v_acc32, v_acc32, v_tmp0);
    str(q_acc32, ptr(reg_c3, 32));
    ldr(q_tmp1, ptr(reg_c3, 48));
    add(v_acc33, v_acc33, v_tmp1);
    str(q_acc33, ptr(reg_c3, 48));

    L(done);
    postamble();
}

jit_int8_brgemm_kernel_4x16_dot_packed_lhs_strided_interleaved4::
    jit_int8_brgemm_kernel_4x16_dot_packed_lhs_strided_interleaved4()
    : jit_generator() {}

void jit_int8_brgemm_kernel_4x16_dot_packed_lhs_strided_interleaved4::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_4x16_dot_packed_lhs_strided_interleaved4::generate() {
    preamble();

    const XReg reg_src = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_src_stride = abi_param5;
    const XReg reg_ldc = abi_param6;
    const XReg reg_accum = abi_param7;

    const XReg reg_src0 = x10;
    const XReg reg_src1 = x11;
    const XReg reg_src2 = x12;
    const XReg reg_src3 = x13;
    const XReg reg_c0 = x14;
    const XReg reg_c1 = x15;
    const XReg reg_c2 = x16;
    const XReg reg_c3 = x17;
    const XReg reg_kb = x18;
    const XReg reg_kb4 = x23;
    const XReg reg_kb_tail = x24;

    const VReg4S v_src0s = VReg4S(0);
    const VReg4S v_src1s = VReg4S(1);
    const VReg4S v_src2s = VReg4S(2);
    const VReg4S v_src3s = VReg4S(3);
    const VReg16B v_src0b(v_src0s.getIdx());
    const VReg16B v_src1b(v_src1s.getIdx());
    const VReg16B v_src2b(v_src2s.getIdx());
    const VReg16B v_src3b(v_src3s.getIdx());
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

    const VReg4S v_tmp0 = VReg4S(8);
    const VReg4S v_tmp1 = VReg4S(9);
    const QReg q_acc00(v_acc00.getIdx());
    const QReg q_acc01(v_acc01.getIdx());
    const QReg q_acc02(v_acc02.getIdx());
    const QReg q_acc03(v_acc03.getIdx());
    const QReg q_acc10(v_acc10.getIdx());
    const QReg q_acc11(v_acc11.getIdx());
    const QReg q_acc12(v_acc12.getIdx());
    const QReg q_acc13(v_acc13.getIdx());
    const QReg q_acc20(v_acc20.getIdx());
    const QReg q_acc21(v_acc21.getIdx());
    const QReg q_acc22(v_acc22.getIdx());
    const QReg q_acc23(v_acc23.getIdx());
    const QReg q_acc30(v_acc30.getIdx());
    const QReg q_acc31(v_acc31.getIdx());
    const QReg q_acc32(v_acc32.getIdx());
    const QReg q_acc33(v_acc33.getIdx());
    const QReg q_tmp0(v_tmp0.getIdx());
    const QReg q_tmp1(v_tmp1.getIdx());

    auto prefetch_next = [&]() {
        prfm(PLDL1KEEP, ptr(reg_wei, 256));
        prfm(PLDL1KEEP, ptr(reg_src0, 64));
        prfm(PLDL1KEEP, ptr(reg_src1, 64));
        prfm(PLDL1KEEP, ptr(reg_src2, 64));
        prfm(PLDL1KEEP, ptr(reg_src3, 64));
    };

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

    mov(reg_src0, reg_src);
    add(reg_src1, reg_src0, reg_src_stride);
    add(reg_src2, reg_src1, reg_src_stride);
    add(reg_src3, reg_src2, reg_src_stride);

    mov(reg_c0, reg_dst);
    add(reg_c1, reg_c0, reg_ldc);
    add(reg_c2, reg_c1, reg_ldc);
    add(reg_c3, reg_c2, reg_ldc);

    lsr(reg_kb, reg_k, 2);

    Label loop4;
    Label loop1;
    Label store_acc;
    Label done;

    cbz(reg_kb, done);

    lsr(reg_kb4, reg_kb, 2);
    lsl(reg_kb_tail, reg_kb4, 2);
    sub(reg_kb, reg_kb, reg_kb_tail);

    cbz(reg_kb4, loop1);

    L(loop4);
    prefetch_next();
    for (int unroll = 0; unroll < 4; ++unroll) {
        ld1(v_src0b, post_ptr(reg_src0, 16));
        ld1(v_src1b, post_ptr(reg_src1, 16));
        ld1(v_src2b, post_ptr(reg_src2, 16));
        ld1(v_src3b, post_ptr(reg_src3, 16));
        ld1(VReg16BList(v_w0, v_w1), post_ptr(reg_wei, 32));
        ld1(VReg16BList(v_w2, v_w3), post_ptr(reg_wei, 32));
        sdot(v_acc00, v_src0b, v_w0);
        sdot(v_acc01, v_src0b, v_w1);
        sdot(v_acc02, v_src0b, v_w2);
        sdot(v_acc03, v_src0b, v_w3);
        sdot(v_acc10, v_src1b, v_w0);
        sdot(v_acc11, v_src1b, v_w1);
        sdot(v_acc12, v_src1b, v_w2);
        sdot(v_acc13, v_src1b, v_w3);
        sdot(v_acc20, v_src2b, v_w0);
        sdot(v_acc21, v_src2b, v_w1);
        sdot(v_acc22, v_src2b, v_w2);
        sdot(v_acc23, v_src2b, v_w3);
        sdot(v_acc30, v_src3b, v_w0);
        sdot(v_acc31, v_src3b, v_w1);
        sdot(v_acc32, v_src3b, v_w2);
        sdot(v_acc33, v_src3b, v_w3);
    }
    subs(reg_kb4, reg_kb4, 1);
    b(NE, loop4);

    L(loop1);
    cbz(reg_kb, done);
    prefetch_next();
    ld1(v_src0b, post_ptr(reg_src0, 16));
    ld1(v_src1b, post_ptr(reg_src1, 16));
    ld1(v_src2b, post_ptr(reg_src2, 16));
    ld1(v_src3b, post_ptr(reg_src3, 16));
    ld1(VReg16BList(v_w0, v_w1), post_ptr(reg_wei, 32));
    ld1(VReg16BList(v_w2, v_w3), post_ptr(reg_wei, 32));
    sdot(v_acc00, v_src0b, v_w0);
    sdot(v_acc01, v_src0b, v_w1);
    sdot(v_acc02, v_src0b, v_w2);
    sdot(v_acc03, v_src0b, v_w3);
    sdot(v_acc10, v_src1b, v_w0);
    sdot(v_acc11, v_src1b, v_w1);
    sdot(v_acc12, v_src1b, v_w2);
    sdot(v_acc13, v_src1b, v_w3);
    sdot(v_acc20, v_src2b, v_w0);
    sdot(v_acc21, v_src2b, v_w1);
    sdot(v_acc22, v_src2b, v_w2);
    sdot(v_acc23, v_src2b, v_w3);
    sdot(v_acc30, v_src3b, v_w0);
    sdot(v_acc31, v_src3b, v_w1);
    sdot(v_acc32, v_src3b, v_w2);
    sdot(v_acc33, v_src3b, v_w3);
    subs(reg_kb, reg_kb, 1);
    b(NE, loop1);

    cbnz(reg_accum, store_acc);
    str(q_acc00, ptr(reg_c0));
    str(q_acc01, ptr(reg_c0, 16));
    str(q_acc02, ptr(reg_c0, 32));
    str(q_acc03, ptr(reg_c0, 48));
    str(q_acc10, ptr(reg_c1));
    str(q_acc11, ptr(reg_c1, 16));
    str(q_acc12, ptr(reg_c1, 32));
    str(q_acc13, ptr(reg_c1, 48));
    str(q_acc20, ptr(reg_c2));
    str(q_acc21, ptr(reg_c2, 16));
    str(q_acc22, ptr(reg_c2, 32));
    str(q_acc23, ptr(reg_c2, 48));
    str(q_acc30, ptr(reg_c3));
    str(q_acc31, ptr(reg_c3, 16));
    str(q_acc32, ptr(reg_c3, 32));
    str(q_acc33, ptr(reg_c3, 48));
    b(done);

    L(store_acc);
    ldr(q_tmp0, ptr(reg_c0));
    add(v_acc00, v_acc00, v_tmp0);
    str(q_acc00, ptr(reg_c0));
    ldr(q_tmp1, ptr(reg_c0, 16));
    add(v_acc01, v_acc01, v_tmp1);
    str(q_acc01, ptr(reg_c0, 16));
    ldr(q_tmp0, ptr(reg_c0, 32));
    add(v_acc02, v_acc02, v_tmp0);
    str(q_acc02, ptr(reg_c0, 32));
    ldr(q_tmp1, ptr(reg_c0, 48));
    add(v_acc03, v_acc03, v_tmp1);
    str(q_acc03, ptr(reg_c0, 48));

    ldr(q_tmp0, ptr(reg_c1));
    add(v_acc10, v_acc10, v_tmp0);
    str(q_acc10, ptr(reg_c1));
    ldr(q_tmp1, ptr(reg_c1, 16));
    add(v_acc11, v_acc11, v_tmp1);
    str(q_acc11, ptr(reg_c1, 16));
    ldr(q_tmp0, ptr(reg_c1, 32));
    add(v_acc12, v_acc12, v_tmp0);
    str(q_acc12, ptr(reg_c1, 32));
    ldr(q_tmp1, ptr(reg_c1, 48));
    add(v_acc13, v_acc13, v_tmp1);
    str(q_acc13, ptr(reg_c1, 48));

    ldr(q_tmp0, ptr(reg_c2));
    add(v_acc20, v_acc20, v_tmp0);
    str(q_acc20, ptr(reg_c2));
    ldr(q_tmp1, ptr(reg_c2, 16));
    add(v_acc21, v_acc21, v_tmp1);
    str(q_acc21, ptr(reg_c2, 16));
    ldr(q_tmp0, ptr(reg_c2, 32));
    add(v_acc22, v_acc22, v_tmp0);
    str(q_acc22, ptr(reg_c2, 32));
    ldr(q_tmp1, ptr(reg_c2, 48));
    add(v_acc23, v_acc23, v_tmp1);
    str(q_acc23, ptr(reg_c2, 48));

    ldr(q_tmp0, ptr(reg_c3));
    add(v_acc30, v_acc30, v_tmp0);
    str(q_acc30, ptr(reg_c3));
    ldr(q_tmp1, ptr(reg_c3, 16));
    add(v_acc31, v_acc31, v_tmp1);
    str(q_acc31, ptr(reg_c3, 16));
    ldr(q_tmp0, ptr(reg_c3, 32));
    add(v_acc32, v_acc32, v_tmp0);
    str(q_acc32, ptr(reg_c3, 32));
    ldr(q_tmp1, ptr(reg_c3, 48));
    add(v_acc33, v_acc33, v_tmp1);
    str(q_acc33, ptr(reg_c3, 48));

    L(done);
    postamble();
}

}  // namespace ov::intel_cpu::aarch64
