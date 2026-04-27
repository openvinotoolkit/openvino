// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/aarch64/brgemm_kernels/int8_brgemm_kernels.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_gen.h>

#include "utils/cpu_utils.hpp"

namespace ov::intel_cpu::aarch64 {

using namespace Xbyak_aarch64;
using namespace dnnl::impl::cpu::aarch64;

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

}  // namespace ov::intel_cpu::aarch64
