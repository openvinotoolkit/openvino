// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/aarch64/brgemm_kernels/int8_brgemm_kernels.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_gen.h>

#include "utils/cpu_utils.hpp"

namespace ov::intel_cpu::aarch64 {

using namespace Xbyak_aarch64;
using namespace dnnl::impl::cpu::aarch64;

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

}  // namespace ov::intel_cpu::aarch64
