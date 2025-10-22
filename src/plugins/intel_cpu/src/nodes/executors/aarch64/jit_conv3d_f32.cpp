// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/aarch64/jit_conv3d_f32.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_gen.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_label.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h>

#include <cstddef>
#include <cstdint>

#include "cpu_memory.h"

using namespace dnnl::impl::cpu::aarch64;

namespace ov::intel_cpu {

void JitConv3DKernelF32::create_ker() {
    jit_generator::create_kernel();
    ker_ = reinterpret_cast<jit_fn>(const_cast<uint8_t*>(jit_ker()));
}

void JitConv3DKernelF32::generate() {
    using namespace Xbyak_aarch64;

    const XReg reg_args = abi_param1;

    const XReg reg_src = x1;
    const XReg reg_wei = x2;
    const XReg reg_wei2 = x3;
    const XReg reg_reps = x4;
    const XReg reg_tail = x5;
    const XReg reg_src_stride = x6;
    const XReg reg_wei_stride = x7;
    const XReg reg_src_blk_stride = x8;
    const XReg reg_wei_blk_stride = x9;
    const XReg reg_acc = x10;
    const XReg reg_acc2 = x11;
    const XReg reg_kw_cnt = x12;
    const XReg reg_src_dx = x13;
    const XReg reg_wei_dx = x14;

    ldr(reg_src, ptr(reg_args, 0));
    ldr(reg_wei, ptr(reg_args, 8));
    ldr(reg_wei2, ptr(reg_args, 16));
    ldr(reg_reps, ptr(reg_args, 24));
    ldr(reg_tail, ptr(reg_args, 32));
    ldr(reg_src_stride, ptr(reg_args, 40));
    ldr(reg_wei_stride, ptr(reg_args, 48));
    ldr(reg_src_blk_stride, ptr(reg_args, 56));
    ldr(reg_wei_blk_stride, ptr(reg_args, 64));
    ldr(reg_acc, ptr(reg_args, 72));
    ldr(reg_acc2, ptr(reg_args, 80));
    ldr(reg_kw_cnt, ptr(reg_args, 88));
    ldr(reg_src_dx, ptr(reg_args, 96));
    ldr(reg_wei_dx, ptr(reg_args, 104));

    const XReg q_src_base = x15;
    const XReg q_wei_base = x16;
    const XReg q_wei2_base = x17;

    Label Lsingle, Ldone;
    Label Ldual_kx, Lkx_d, Ltail_prep_d_kx, Ltail_done_d_kx;
    Label Lsingle_kx, Lkx_s, Ltail_prep_s_kx, Ltail_done_s_kx;

    cbz(reg_acc2, Lsingle);
    b(Ldual_kx);

    L(Ldual_kx);
    eor(VReg16B(20), VReg16B(20), VReg16B(20));
    eor(VReg16B(21), VReg16B(21), VReg16B(21));

    mov(q_src_base, reg_src);
    mov(q_wei_base, reg_wei);
    mov(q_wei2_base, reg_wei2);
    cbnz(reg_kw_cnt, Lkx_d);
    mov(reg_kw_cnt, 1);

    L(Lkx_d);
    ldr(reg_reps, ptr(reg_args, 24));
    mov(reg_src, q_src_base);
    mov(reg_wei, q_wei_base);
    mov(reg_wei2, q_wei2_base);

    Label Lrep_d;
    L(Lrep_d);
    cmp(reg_reps, 0);
    b(EQ, Ltail_prep_d_kx);
    ld1(VReg(0).s[0], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[1], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[2], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[3], ptr(reg_src));
    Label Lw_np_d, Lw_done_d;
    cmp(reg_wei_stride, 4);
    b(NE, Lw_np_d);
    ld1(VReg4S(1), ptr(reg_wei));
    ld1(VReg4S(2), ptr(reg_wei2));
    add(reg_wei, reg_wei, reg_wei_blk_stride);
    add(reg_wei2, reg_wei2, reg_wei_blk_stride);
    b(Lw_done_d);
    L(Lw_np_d);
    ld1(VReg(1).s[0], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(2).s[0], ptr(reg_wei2));
    add(reg_wei2, reg_wei2, reg_wei_stride);
    ld1(VReg(1).s[1], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(2).s[1], ptr(reg_wei2));
    add(reg_wei2, reg_wei2, reg_wei_stride);
    ld1(VReg(1).s[2], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(2).s[2], ptr(reg_wei2));
    add(reg_wei2, reg_wei2, reg_wei_stride);
    ld1(VReg(1).s[3], ptr(reg_wei));
    ld1(VReg(2).s[3], ptr(reg_wei2));
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    L(Lw_done_d);
    add(reg_src, reg_src, reg_src_stride);
    fmla(VReg4S(20), VReg4S(0), VReg4S(1));
    fmla(VReg4S(21), VReg4S(0), VReg4S(2));
    sub(reg_reps, reg_reps, 1);
    b(Lrep_d);

    L(Ltail_prep_d_kx);
    eor(VReg16B(0), VReg16B(0), VReg16B(0));
    eor(VReg16B(1), VReg16B(1), VReg16B(1));
    eor(VReg16B(2), VReg16B(2), VReg16B(2));
    cmp(reg_tail, 0);
    b(LE, Ltail_done_d_kx);
    ld1(VReg(0).s[0], ptr(reg_src));
    ld1(VReg(1).s[0], ptr(reg_wei));
    ld1(VReg(2).s[0], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    cmp(reg_tail, 1);
    b(LE, Ltail_done_d_kx);
    ld1(VReg(0).s[1], ptr(reg_src));
    ld1(VReg(1).s[1], ptr(reg_wei));
    ld1(VReg(2).s[1], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    cmp(reg_tail, 2);
    b(LE, Ltail_done_d_kx);
    ld1(VReg(0).s[2], ptr(reg_src));
    ld1(VReg(1).s[2], ptr(reg_wei));
    ld1(VReg(2).s[2], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    cmp(reg_tail, 3);
    b(LE, Ltail_done_d_kx);
    ld1(VReg(0).s[3], ptr(reg_src));
    ld1(VReg(1).s[3], ptr(reg_wei));
    ld1(VReg(2).s[3], ptr(reg_wei2));
    L(Ltail_done_d_kx);
    fmla(VReg4S(20), VReg4S(0), VReg4S(1));
    fmla(VReg4S(21), VReg4S(0), VReg4S(2));
    sub(reg_kw_cnt, reg_kw_cnt, 1);
    add(q_src_base, q_src_base, reg_src_dx);
    add(q_wei_base, q_wei_base, reg_wei_dx);
    add(q_wei2_base, q_wei2_base, reg_wei_dx);
    cbnz(reg_kw_cnt, Lkx_d);
    faddp(VReg4S(20), VReg4S(20), VReg4S(20));
    faddp(VReg2S(20), VReg2S(20), VReg2S(20));
    faddp(VReg4S(21), VReg4S(21), VReg4S(21));
    faddp(VReg2S(21), VReg2S(21), VReg2S(21));
    ldr(SReg(0), ptr(reg_acc));
    fadd(SReg(0), SReg(0), SReg(20));
    str(SReg(0), ptr(reg_acc));
    ldr(SReg(1), ptr(reg_acc2));
    fadd(SReg(1), SReg(1), SReg(21));
    str(SReg(1), ptr(reg_acc2));
    b(Ldone);

    L(Lsingle);
    eor(VReg16B(20), VReg16B(20), VReg16B(20));
    mov(q_src_base, reg_src);
    mov(q_wei_base, reg_wei);
    cbnz(reg_kw_cnt, Lsingle_kx);
    mov(reg_kw_cnt, 1);

    L(Lsingle_kx);
    ldr(reg_reps, ptr(reg_args, 24));
    mov(reg_src, q_src_base);
    mov(reg_wei, q_wei_base);

    Label Lrep_s;
    L(Lrep_s);
    cmp(reg_reps, 0);
    b(EQ, Ltail_prep_s_kx);
    ld1(VReg(0).s[0], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[1], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[2], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[3], ptr(reg_src));
    Label Lw_np_s, Lw_done_s;
    cmp(reg_wei_stride, 4);
    b(NE, Lw_np_s);
    ld1(VReg4S(1), ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_blk_stride);
    b(Lw_done_s);
    L(Lw_np_s);
    ld1(VReg(1).s[0], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(1).s[1], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(1).s[2], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(1).s[3], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    L(Lw_done_s);
    add(reg_src, reg_src, reg_src_stride);
    fmla(VReg4S(20), VReg4S(0), VReg4S(1));
    sub(reg_reps, reg_reps, 1);
    b(Lrep_s);

    L(Ltail_prep_s_kx);
    eor(VReg16B(0), VReg16B(0), VReg16B(0));
    eor(VReg16B(1), VReg16B(1), VReg16B(1));
    cmp(reg_tail, 0);
    b(LE, Ltail_done_s_kx);
    ld1(VReg(0).s[0], ptr(reg_src));
    ld1(VReg(1).s[0], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    cmp(reg_tail, 1);
    b(LE, Ltail_done_s_kx);
    ld1(VReg(0).s[1], ptr(reg_src));
    ld1(VReg(1).s[1], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    cmp(reg_tail, 2);
    b(LE, Ltail_done_s_kx);
    ld1(VReg(0).s[2], ptr(reg_src));
    ld1(VReg(1).s[2], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    cmp(reg_tail, 3);
    b(LE, Ltail_done_s_kx);
    ld1(VReg(0).s[3], ptr(reg_src));
    ld1(VReg(1).s[3], ptr(reg_wei));
    L(Ltail_done_s_kx);
    fmla(VReg4S(20), VReg4S(0), VReg4S(1));

    sub(reg_kw_cnt, reg_kw_cnt, 1);
    add(q_src_base, q_src_base, reg_src_dx);
    add(q_wei_base, q_wei_base, reg_wei_dx);
    cbnz(reg_kw_cnt, Lsingle_kx);

    faddp(VReg4S(20), VReg4S(20), VReg4S(20));
    faddp(VReg2S(20), VReg2S(20), VReg2S(20));
    ldr(SReg(0), ptr(reg_acc));
    fadd(SReg(0), SReg(0), SReg(20));
    str(SReg(0), ptr(reg_acc));
    b(Ldone);

    L(Ldone);
    ret();
}

}  // namespace ov::intel_cpu
