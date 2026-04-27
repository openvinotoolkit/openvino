// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/aarch64/brgemm_kernels/int8_brgemm_kernels.hpp"

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

}  // namespace ov::intel_cpu::aarch64

