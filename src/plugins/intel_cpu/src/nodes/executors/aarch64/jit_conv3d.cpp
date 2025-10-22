// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/aarch64/jit_conv3d.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_gen.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_label.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h>

#include <algorithm>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "cpu_memory.h"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/cpu_utils.hpp"

using namespace dnnl::impl::cpu::aarch64;

namespace ov::intel_cpu {

JitConv3DKernelF16::JitConv3DKernelF16() = default;

void JitConv3DKernelF16::create_ker() {
    jit_generator::create_kernel();
    ker_ = jit_kernel_cast<jit_fn>(jit_ker());
}

void JitConv3DKernelF16::gen_minimal_kernel() {
    using namespace Xbyak_aarch64;
    const XReg reg_args = abi_param1;  // x0
    const XReg reg_src = x1;         // const uint16_t* src
    const XReg reg_wei = x2;         // const uint16_t* wei
    const XReg reg_wei2 = x3;        // const uint16_t* wei2 (optional)
    const XReg reg_reps = x4;        // size_t repeats
    const XReg reg_tail = x5;        // size_t tail
    const XReg reg_src_stride = x6;  // size_t src_stride (bytes)
    const XReg reg_wei_stride = x7;  // size_t wei_stride (bytes)
    const XReg reg_acc = x8;         // float* acc
    const XReg reg_acc2 = x9;        // float* acc2 (optional)

    ldr(reg_src, ptr(reg_args, 0));
    ldr(reg_wei, ptr(reg_args, 8));
    ldr(reg_wei2, ptr(reg_args, 16));
    ldr(reg_reps, ptr(reg_args, 40));
    ldr(reg_tail, ptr(reg_args, 48));
    ldr(reg_src_stride, ptr(reg_args, 56));
    ldr(reg_wei_stride, ptr(reg_args, 64));
    ldr(reg_acc, ptr(reg_args, 88));
    ldr(reg_acc2, ptr(reg_args, 96));

    Label Lsingle, Ldone;
    Label Ldual_kx, Lkx_d, Ltail_prep_d_kx, Ltail_done_d_kx;
    Label Lsingle_kx, Lkx_s, Ltail_prep_s_kx, Ltail_done_s_kx;
    cbz(reg_acc2, Lsingle);
    b(Ldual_kx);

    L(Ldual_kx);
    eor(VReg16B(20), VReg16B(20), VReg16B(20));
    eor(VReg16B(21), VReg16B(21), VReg16B(21));
    const XReg reg_kw_cnt = x12;
    const XReg reg_src_dx = x13;
    const XReg reg_wei_dx = x14;
    ldr(reg_kw_cnt, ptr(reg_args, 104));
    ldr(reg_src_dx, ptr(reg_args, 112));
    ldr(reg_wei_dx, ptr(reg_args, 120));
    const XReg q_src_base = x15;
    const XReg q_wei_base = x16;
    const XReg q_wei2_base = x17;
    const XReg reg_wei_blk_stride2 = x10;
    ldr(reg_wei_blk_stride2, ptr(reg_args, 80));
    mov(q_src_base, reg_src);
    mov(q_wei_base, reg_wei);
    mov(q_wei2_base, reg_wei2);
    cbnz(reg_kw_cnt, Lkx_d);
    mov(reg_kw_cnt, 1);
    L(Lkx_d);
    ldr(reg_reps, ptr(reg_args, 40));
    mov(reg_src, q_src_base);
    mov(reg_wei, q_wei_base);
    mov(reg_wei2, q_wei2_base);
    Label Lrep_d_kx;
    L(Lrep_d_kx);
    cmp(reg_reps, 0);
    b(EQ, Ltail_prep_d_kx);
    ld1(VReg(0).h[0], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[1], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[2], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[3], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[4], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[5], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[6], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[7], ptr(reg_src));
    // Load weights for oc0/oc1 (vector fast path if stride==2)
    Label Lw_np_d, Lw_done_d2;
    cmp(reg_wei_stride, 2);
    b(NE, Lw_np_d);
    ld1(VReg8H(1), ptr(reg_wei));
    ld1(VReg8H(2), ptr(reg_wei2));
    add(reg_wei, reg_wei, reg_wei_blk_stride2);
    add(reg_wei2, reg_wei2, reg_wei_blk_stride2);
    b(Lw_done_d2);
    L(Lw_np_d);
    ld1(VReg(1).h[0], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(2).h[0], ptr(reg_wei2));
    add(reg_wei2, reg_wei2, reg_wei_stride);
    ld1(VReg(1).h[1], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(2).h[1], ptr(reg_wei2));
    add(reg_wei2, reg_wei2, reg_wei_stride);
    ld1(VReg(1).h[2], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(2).h[2], ptr(reg_wei2));
    add(reg_wei2, reg_wei2, reg_wei_stride);
    ld1(VReg(1).h[3], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(2).h[3], ptr(reg_wei2));
    add(reg_wei2, reg_wei2, reg_wei_stride);
    ld1(VReg(1).h[4], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(2).h[4], ptr(reg_wei2));
    add(reg_wei2, reg_wei2, reg_wei_stride);
    ld1(VReg(1).h[5], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(2).h[5], ptr(reg_wei2));
    add(reg_wei2, reg_wei2, reg_wei_stride);
    ld1(VReg(1).h[6], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(2).h[6], ptr(reg_wei2));
    add(reg_wei2, reg_wei2, reg_wei_stride);
    ld1(VReg(1).h[7], ptr(reg_wei));
    ld1(VReg(2).h[7], ptr(reg_wei2));
    L(Lw_done_d2);
    fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
    fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
    fmlal(VReg4S(21), VReg4H(0), VReg4H(2));
    fmlal2(VReg4S(21), VReg4H(0), VReg4H(2));
    sub(reg_reps, reg_reps, 1);
    b(Lrep_d_kx);
    L(Ltail_prep_d_kx);
    eor(VReg16B(0), VReg16B(0), VReg16B(0));
    eor(VReg16B(1), VReg16B(1), VReg16B(1));
    eor(VReg16B(2), VReg16B(2), VReg16B(2));
    cmp(reg_tail, 0);
    b(LE, Ltail_done_d_kx);
    ld1(VReg(0).h[0], ptr(reg_src));
    ld1(VReg(1).h[0], ptr(reg_wei));
    ld1(VReg(2).h[0], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    cmp(reg_tail, 1);
    b(LE, Ltail_done_d_kx);
    ld1(VReg(0).h[1], ptr(reg_src));
    ld1(VReg(1).h[1], ptr(reg_wei));
    ld1(VReg(2).h[1], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    cmp(reg_tail, 2);
    b(LE, Ltail_done_d_kx);
    ld1(VReg(0).h[2], ptr(reg_src));
    ld1(VReg(1).h[2], ptr(reg_wei));
    ld1(VReg(2).h[2], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    cmp(reg_tail, 3);
    b(LE, Ltail_done_d_kx);
    ld1(VReg(0).h[3], ptr(reg_src));
    ld1(VReg(1).h[3], ptr(reg_wei));
    ld1(VReg(2).h[3], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    cmp(reg_tail, 4);
    b(LE, Ltail_done_d_kx);
    ld1(VReg(0).h[4], ptr(reg_src));
    ld1(VReg(1).h[4], ptr(reg_wei));
    ld1(VReg(2).h[4], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    cmp(reg_tail, 5);
    b(LE, Ltail_done_d_kx);
    ld1(VReg(0).h[5], ptr(reg_src));
    ld1(VReg(1).h[5], ptr(reg_wei));
    ld1(VReg(2).h[5], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    cmp(reg_tail, 6);
    b(LE, Ltail_done_d_kx);
    ld1(VReg(0).h[6], ptr(reg_src));
    ld1(VReg(1).h[6], ptr(reg_wei));
    ld1(VReg(2).h[6], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    cmp(reg_tail, 7);
    b(LE, Ltail_done_d_kx);
    ld1(VReg(0).h[7], ptr(reg_src));
    ld1(VReg(1).h[7], ptr(reg_wei));
    ld1(VReg(2).h[7], ptr(reg_wei2));
    L(Ltail_done_d_kx);
    fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
    fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
    fmlal(VReg4S(21), VReg4H(0), VReg4H(2));
    fmlal2(VReg4S(21), VReg4H(0), VReg4H(2));
    sub(reg_kw_cnt, reg_kw_cnt, 1);
    add(q_src_base, q_src_base, reg_src_dx);
    add(q_wei_base, q_wei_base, reg_wei_dx);
    add(q_wei2_base, q_wei2_base, reg_wei_dx);
    cbnz(reg_kw_cnt, Lkx_d);
    // reduce and store accumulators
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

    // Dual-OC path: v20 (oc0), v21 (oc1)
    eor(VReg16B(20), VReg16B(20), VReg16B(20));
    eor(VReg16B(21), VReg16B(21), VReg16B(21));

    Label Lrep_d, Ltail_prep_d, Ltail_done_d;
    L(Lrep_d);
    cmp(reg_reps, 0);
    b(EQ, Ltail_prep_d);
    ld1(VReg(0).h[0], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[1], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[2], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[3], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[4], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[5], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[6], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[7], ptr(reg_src));
    // Load wei lanes for oc0 (v1) and oc1 (v2) — vector fast path if wei_stride==2
    Label Ldw_np_d, Ldw_done_d;
    cmp(reg_wei_stride, 2);
    b(NE, Ldw_np_d);
    ld1(VReg8H(1), ptr(reg_wei));
    ld1(VReg8H(2), ptr(reg_wei2));
    add(reg_wei, reg_wei, 16);
    add(reg_wei2, reg_wei2, 16);
    b(Ldw_done_d);
    L(Ldw_np_d);
    ld1(VReg(1).h[0], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(2).h[0], ptr(reg_wei2));
    add(reg_wei2, reg_wei2, reg_wei_stride);
    ld1(VReg(1).h[1], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(2).h[1], ptr(reg_wei2));
    add(reg_wei2, reg_wei2, reg_wei_stride);
    ld1(VReg(1).h[2], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(2).h[2], ptr(reg_wei2));
    add(reg_wei2, reg_wei2, reg_wei_stride);
    ld1(VReg(1).h[3], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(2).h[3], ptr(reg_wei2));
    add(reg_wei2, reg_wei2, reg_wei_stride);
    ld1(VReg(1).h[4], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(2).h[4], ptr(reg_wei2));
    add(reg_wei2, reg_wei2, reg_wei_stride);
    ld1(VReg(1).h[5], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(2).h[5], ptr(reg_wei2));
    add(reg_wei2, reg_wei2, reg_wei_stride);
    ld1(VReg(1).h[6], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(2).h[6], ptr(reg_wei2));
    add(reg_wei2, reg_wei2, reg_wei_stride);
    ld1(VReg(1).h[7], ptr(reg_wei));
    ld1(VReg(2).h[7], ptr(reg_wei2));
    L(Ldw_done_d);
    // MAC into v20/v21
    fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
    fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
    fmlal(VReg4S(21), VReg4H(0), VReg4H(2));
    fmlal2(VReg4S(21), VReg4H(0), VReg4H(2));
    sub(reg_reps, reg_reps, 1);
    b(Lrep_d);

    L(Ltail_prep_d);
    eor(VReg16B(0), VReg16B(0), VReg16B(0));
    eor(VReg16B(1), VReg16B(1), VReg16B(1));
    eor(VReg16B(2), VReg16B(2), VReg16B(2));
    // lanes 0..7 guarded by tail
    cmp(reg_tail, 0);
    b(LE, Ltail_done_d);
    ld1(VReg(0).h[0], ptr(reg_src));
    ld1(VReg(1).h[0], ptr(reg_wei));
    ld1(VReg(2).h[0], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    cmp(reg_tail, 1);
    b(LE, Ltail_done_d);
    ld1(VReg(0).h[1], ptr(reg_src));
    ld1(VReg(1).h[1], ptr(reg_wei));
    ld1(VReg(2).h[1], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    cmp(reg_tail, 2);
    b(LE, Ltail_done_d);
    ld1(VReg(0).h[2], ptr(reg_src));
    ld1(VReg(1).h[2], ptr(reg_wei));
    ld1(VReg(2).h[2], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    cmp(reg_tail, 3);
    b(LE, Ltail_done_d);
    ld1(VReg(0).h[3], ptr(reg_src));
    ld1(VReg(1).h[3], ptr(reg_wei));
    ld1(VReg(2).h[3], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    cmp(reg_tail, 4);
    b(LE, Ltail_done_d);
    ld1(VReg(0).h[4], ptr(reg_src));
    ld1(VReg(1).h[4], ptr(reg_wei));
    ld1(VReg(2).h[4], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    cmp(reg_tail, 5);
    b(LE, Ltail_done_d);
    ld1(VReg(0).h[5], ptr(reg_src));
    ld1(VReg(1).h[5], ptr(reg_wei));
    ld1(VReg(2).h[5], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    cmp(reg_tail, 6);
    b(LE, Ltail_done_d);
    ld1(VReg(0).h[6], ptr(reg_src));
    ld1(VReg(1).h[6], ptr(reg_wei));
    ld1(VReg(2).h[6], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    cmp(reg_tail, 7);
    b(LE, Ltail_done_d);
    ld1(VReg(0).h[7], ptr(reg_src));
    ld1(VReg(1).h[7], ptr(reg_wei));
    ld1(VReg(2).h[7], ptr(reg_wei2));

    L(Ltail_done_d);
    fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
    fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
    fmlal(VReg4S(21), VReg4H(0), VReg4H(2));
    fmlal2(VReg4S(21), VReg4H(0), VReg4H(2));
    // horizontal add and store
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

    // Single-OC path
    L(Lsingle);
    b(Lsingle_kx);
    // Single-OC with in-kernel kx loop
    L(Lsingle_kx);
    eor(VReg16B(20), VReg16B(20), VReg16B(20));
    const XReg s_kw_cnt = x12;
    const XReg s_src_dx = x13;
    const XReg s_wei_dx = x14;
    ldr(s_kw_cnt, ptr(reg_args, 104));
    ldr(s_src_dx, ptr(reg_args, 112));
    ldr(s_wei_dx, ptr(reg_args, 120));
    const XReg s_src_base = x15;
    const XReg s_wei_base = x16;
    const XReg s_wei_blk_stride2 = x10;
    ldr(s_wei_blk_stride2, ptr(reg_args, 80));
    mov(s_src_base, reg_src);
    mov(s_wei_base, reg_wei);
    cbnz(s_kw_cnt, Lkx_s);
    mov(s_kw_cnt, 1);
    Label Lrep_s_kx;
    L(Lkx_s);
    ldr(reg_reps, ptr(reg_args, 40));
    mov(reg_src, s_src_base);
    mov(reg_wei, s_wei_base);
    L(Lrep_s_kx);
    cmp(reg_reps, 0);
    b(EQ, Ltail_prep_s_kx);
    ld1(VReg(0).h[0], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[1], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[2], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[3], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[4], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[5], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[6], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[7], ptr(reg_src));
    // weights (vector fast path if stride==2)
    Label Lw_np_s, Lw_done_s2;
    cmp(reg_wei_stride, 2);
    b(NE, Lw_np_s);
    ld1(VReg8H(1), ptr(reg_wei));
    add(reg_wei, reg_wei, s_wei_blk_stride2);
    b(Lw_done_s2);
    L(Lw_np_s);
    ld1(VReg(1).h[0], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(1).h[1], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(1).h[2], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(1).h[3], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(1).h[4], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(1).h[5], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(1).h[6], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(1).h[7], ptr(reg_wei));
    L(Lw_done_s2);
    fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
    fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
    sub(reg_reps, reg_reps, 1);
    b(Lrep_s_kx);
    L(Ltail_prep_s_kx);
    eor(VReg16B(0), VReg16B(0), VReg16B(0));
    eor(VReg16B(1), VReg16B(1), VReg16B(1));
    cmp(reg_tail, 0);
    b(LE, Ltail_done_s_kx);
    ld1(VReg(0).h[0], ptr(reg_src));
    ld1(VReg(1).h[0], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    cmp(reg_tail, 1);
    b(LE, Ltail_done_s_kx);
    ld1(VReg(0).h[1], ptr(reg_src));
    ld1(VReg(1).h[1], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    cmp(reg_tail, 2);
    b(LE, Ltail_done_s_kx);
    ld1(VReg(0).h[2], ptr(reg_src));
    ld1(VReg(1).h[2], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    cmp(reg_tail, 3);
    b(LE, Ltail_done_s_kx);
    ld1(VReg(0).h[3], ptr(reg_src));
    ld1(VReg(1).h[3], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    cmp(reg_tail, 4);
    b(LE, Ltail_done_s_kx);
    ld1(VReg(0).h[4], ptr(reg_src));
    ld1(VReg(1).h[4], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    cmp(reg_tail, 5);
    b(LE, Ltail_done_s_kx);
    ld1(VReg(0).h[5], ptr(reg_src));
    ld1(VReg(1).h[5], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    cmp(reg_tail, 6);
    b(LE, Ltail_done_s_kx);
    ld1(VReg(0).h[6], ptr(reg_src));
    ld1(VReg(1).h[6], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    cmp(reg_tail, 7);
    b(LE, Ltail_done_s_kx);
    ld1(VReg(0).h[7], ptr(reg_src));
    ld1(VReg(1).h[7], ptr(reg_wei));
    L(Ltail_done_s_kx);
    fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
    fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
    sub(s_kw_cnt, s_kw_cnt, 1);
    add(s_src_base, s_src_base, s_src_dx);
    add(s_wei_base, s_wei_base, s_wei_dx);
    cbnz(s_kw_cnt, Lkx_s);
    // reduce/store
    faddp(VReg4S(20), VReg4S(20), VReg4S(20));
    faddp(VReg2S(20), VReg2S(20), VReg2S(20));
    ldr(SReg(0), ptr(reg_acc));
    fadd(SReg(0), SReg(0), SReg(20));
    str(SReg(0), ptr(reg_acc));
    eor(VReg16B(20), VReg16B(20), VReg16B(20));
    Label Lrep_s, Ltail_prep_s, Ltail_done_s;
    L(Lrep_s);
    cmp(reg_reps, 0);
    b(EQ, Ltail_prep_s);
    // src lanes
    ld1(VReg(0).h[0], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[1], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[2], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[3], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[4], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[5], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[6], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[7], ptr(reg_src));
    // wei lanes — vector fast path if wei_stride==2
    Label Ldw_np_s, Ldw_done_s;
    cmp(reg_wei_stride, 2);
    b(NE, Ldw_np_s);
    ld1(VReg8H(1), ptr(reg_wei));
    add(reg_wei, reg_wei, 16);
    b(Ldw_done_s);
    L(Ldw_np_s);
    ld1(VReg(1).h[0], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(1).h[1], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(1).h[2], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(1).h[3], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(1).h[4], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(1).h[5], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(1).h[6], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(1).h[7], ptr(reg_wei));
    L(Ldw_done_s);
    fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
    fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
    sub(reg_reps, reg_reps, 1);
    b(Lrep_s);

    // Tail (single)
    L(Ltail_prep_s);
    eor(VReg16B(0), VReg16B(0), VReg16B(0));
    eor(VReg16B(1), VReg16B(1), VReg16B(1));
    cmp(reg_tail, 0);
    b(LE, Ltail_done_s);
    ld1(VReg(0).h[0], ptr(reg_src));
    ld1(VReg(1).h[0], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    cmp(reg_tail, 1);
    b(LE, Ltail_done_s);
    ld1(VReg(0).h[1], ptr(reg_src));
    ld1(VReg(1).h[1], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    cmp(reg_tail, 2);
    b(LE, Ltail_done_s);
    ld1(VReg(0).h[2], ptr(reg_src));
    ld1(VReg(1).h[2], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    cmp(reg_tail, 3);
    b(LE, Ltail_done_s);
    ld1(VReg(0).h[3], ptr(reg_src));
    ld1(VReg(1).h[3], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    cmp(reg_tail, 4);
    b(LE, Ltail_done_s);
    ld1(VReg(0).h[4], ptr(reg_src));
    ld1(VReg(1).h[4], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    cmp(reg_tail, 5);
    b(LE, Ltail_done_s);
    ld1(VReg(0).h[5], ptr(reg_src));
    ld1(VReg(1).h[5], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    cmp(reg_tail, 6);
    b(LE, Ltail_done_s);
    ld1(VReg(0).h[6], ptr(reg_src));
    ld1(VReg(1).h[6], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    cmp(reg_tail, 7);
    b(LE, Ltail_done_s);
    ld1(VReg(0).h[7], ptr(reg_src));
    ld1(VReg(1).h[7], ptr(reg_wei));
    L(Ltail_done_s);
    fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
    fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
    faddp(VReg4S(20), VReg4S(20), VReg4S(20));
    faddp(VReg2S(20), VReg2S(20), VReg2S(20));
    ldr(SReg(0), ptr(reg_acc));
    fadd(SReg(0), SReg(0), SReg(20));
    str(SReg(0), ptr(reg_acc));

    L(Ldone);
    ret();
}

void JitConv3DKernelF16::gen_optimized_kernel() {
    using namespace Xbyak_aarch64;
    // abi_param1 -> args pointer
    const XReg reg_args = abi_param1;
    // Use call-clobbered registers to avoid saving/restoring callee-saved regs
    const XReg reg_src = x1;
    const XReg reg_wei = x2;
    const XReg reg_wei2 = x10;
    const XReg reg_reps = x3;  // number of full 8-lane blocks
    const XReg reg_tail = x9;  // remaining channels (< 8)
    const XReg reg_src_stride = x4;
    const XReg reg_wei_stride = x5;
    const XReg reg_acc = x6;
    const XReg reg_acc2 = x11;
    const XReg reg_wei3 = x19;
    const XReg reg_wei4 = x20;
    const XReg reg_acc3 = x21;
    const XReg reg_acc4 = x22;

    // Prolog: save callee-saved we will use (x19-x29) and LR (x30)
    stp(XReg(19), XReg(20), pre_ptr(sp, -16));
    stp(XReg(21), XReg(22), pre_ptr(sp, -16));
    stp(XReg(23), XReg(24), pre_ptr(sp, -16));
    stp(XReg(25), XReg(26), pre_ptr(sp, -16));
    stp(XReg(27), XReg(28), pre_ptr(sp, -16));
    stp(XReg(29), XReg(30), pre_ptr(sp, -16));

    // Load args
    ldr(reg_src, ptr(reg_args));             // src
    ldr(reg_wei, ptr(reg_args, 8));          // wei
    ldr(reg_wei2, ptr(reg_args, 16));        // wei2 (optional)
    ldr(reg_wei3, ptr(reg_args, 24));        // wei3 (optional)
    ldr(reg_wei4, ptr(reg_args, 32));        // wei4 (optional)
    ldr(reg_reps, ptr(reg_args, 40));        // repeats
    ldr(reg_tail, ptr(reg_args, 48));        // tail (<= 8)
    ldr(reg_src_stride, ptr(reg_args, 56));  // src_stride bytes
    ldr(reg_wei_stride, ptr(reg_args, 64));  // wei_stride bytes
    const XReg reg_src_blk_stride = x7;
    const XReg reg_wei_blk_stride = x8;
    ldr(reg_src_blk_stride, ptr(reg_args, 72));  // src_blk_stride bytes
    ldr(reg_wei_blk_stride, ptr(reg_args, 80));  // wei_blk_stride bytes
    ldr(reg_acc, ptr(reg_args, 88));             // acc (float*)
    ldr(reg_acc2, ptr(reg_args, 96));            // acc2 (float* or 0)
    const XReg reg_kw_cnt = x12;
    const XReg reg_src_dx = x13;
    const XReg reg_wei_dx = x14;
    ldr(reg_kw_cnt, ptr(reg_args, 104));  // kw count (for stride=1 fast path); 0 -> disabled
    ldr(reg_src_dx, ptr(reg_args, 112));  // src dx step in bytes (x dimension)
    ldr(reg_wei_dx, ptr(reg_args, 120));  // wei dx step in bytes (x dimension)
    ldr(reg_acc3, ptr(reg_args, 128));    // acc3 (float* or 0)
    ldr(reg_acc4, ptr(reg_args, 136));    // acc4 (float* or 0)
    const XReg reg_kh_cnt = x26;
    const XReg reg_src_dy = x27;
    const XReg reg_wei_dy = x28;
    ldr(reg_kh_cnt, ptr(reg_args, 144));  // kh count (for stride=1 fast path); 0 -> disabled
    ldr(reg_src_dy, ptr(reg_args, 152));  // src dy step in bytes (y dimension)
    ldr(reg_wei_dy, ptr(reg_args, 160));  // wei dy step in bytes (y dimension)

    // Optionally force single-ky iteration for stability on certain platforms
    if (m_force_single_kh_) {
        mov(reg_kh_cnt, 1);
    }
    eor(reg_acc4, reg_acc4, reg_acc4);

    Label Lsingle, Lend_all;
    // If acc4 != 0, run quad-OC; else if acc2 != 0, run dual-OC; else single-OC.
    Label Lq_entry;
    cbnz(reg_acc4, Lq_entry);
    cbz(reg_acc2, Lsingle);

    // ---------------- Quad-OC path ----------------
    L(Lq_entry);
    {
        // Zero v20..v23
        eor(VReg16B(20), VReg16B(20), VReg16B(20));
        eor(VReg16B(21), VReg16B(21), VReg16B(21));
        eor(VReg16B(22), VReg16B(22), VReg16B(22));
        eor(VReg16B(23), VReg16B(23), VReg16B(23));

        Label Lq_ky_loop, Lq_kx_loop, Lq_loop, Lq_after_loop, Lq_after_fill, Lq_after_kx, Lq_np;
        // Packed fast path if wei_stride == 2
        cmp(reg_wei_stride, 2);
        b(NE, Lq_np);

        // Save repeats and bases for kw loop
        const XReg q_reps_init = x15;
        const XReg q_src_base = x16;
        const XReg q_wei_base = x17;
        // avoid x18 on Apple; use callee-saved and restore at epilog
        const XReg q_wei2_base = x23;
        const XReg q_wei3_base = x24;
        const XReg q_wei4_base = x25;
        const XReg q_kw_init = x28;
        const XReg q_kh_work = x29;
        mov(q_reps_init, reg_reps);
        mov(q_src_base, reg_src);
        mov(q_wei_base, reg_wei);
        mov(q_wei2_base, reg_wei2);
        mov(q_wei3_base, reg_wei3);
        mov(q_wei4_base, reg_wei4);
        mov(q_kw_init, reg_kw_cnt);
        mov(q_kh_work, reg_kh_cnt);

        // ky loop
        L(Lq_ky_loop);
        // kx loop entry
        L(Lq_kx_loop);
        mov(reg_reps, q_reps_init);
        mov(reg_src, q_src_base);
        mov(reg_wei, q_wei_base);
        mov(reg_wei2, q_wei2_base);
        mov(reg_wei3, q_wei3_base);
        mov(reg_wei4, q_wei4_base);
        mov(reg_kw_cnt, q_kw_init);

        // Main repeats loop
        L(Lq_loop);
        cmp(reg_reps, 0);
        b(LE, Lq_after_loop);
        // Load 8 src lanes
        ld1(VReg(0).h[0], ptr(reg_src));
        add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[1], ptr(reg_src));
        add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[2], ptr(reg_src));
        add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[3], ptr(reg_src));
        add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[4], ptr(reg_src));
        add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[5], ptr(reg_src));
        add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[6], ptr(reg_src));
        add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[7], ptr(reg_src));
        // Load 4 weight vectors
        ld1(VReg8H(1), ptr(reg_wei));
        ld1(VReg8H(2), ptr(reg_wei2));
        ld1(VReg8H(3), ptr(reg_wei3));
        ld1(VReg8H(4), ptr(reg_wei4));
        // Accumulate into v20..v23
        fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal(VReg4S(21), VReg4H(0), VReg4H(2));
        fmlal2(VReg4S(21), VReg4H(0), VReg4H(2));
        fmlal(VReg4S(22), VReg4H(0), VReg4H(3));
        fmlal2(VReg4S(22), VReg4H(0), VReg4H(3));
        fmlal(VReg4S(23), VReg4H(0), VReg4H(4));
        fmlal2(VReg4S(23), VReg4H(0), VReg4H(4));
        // Advance block pointers (src already advanced by 8*src_stride during lane loads)
        add(reg_wei, reg_wei, reg_wei_blk_stride);
        add(reg_wei2, reg_wei2, reg_wei_blk_stride);
        add(reg_wei3, reg_wei3, reg_wei_blk_stride);
        add(reg_wei4, reg_wei4, reg_wei_blk_stride);
        sub(reg_reps, reg_reps, 1);
        b(Lq_loop);

        // Tail <=8
        L(Lq_after_loop);
        eor(VReg16B(0), VReg16B(0), VReg16B(0));
        eor(VReg16B(1), VReg16B(1), VReg16B(1));
        eor(VReg16B(2), VReg16B(2), VReg16B(2));
        eor(VReg16B(3), VReg16B(3), VReg16B(3));
        eor(VReg16B(4), VReg16B(4), VReg16B(4));
        cmp(reg_tail, 0);
        b(LE, Lq_after_fill);
        ld1(VReg(0).h[0], ptr(reg_src));
        ld1(VReg(1).h[0], ptr(reg_wei));
        ld1(VReg(2).h[0], ptr(reg_wei2));
        ld1(VReg(3).h[0], ptr(reg_wei3));
        ld1(VReg(4).h[0], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        add(reg_wei3, reg_wei3, reg_wei_stride);
        add(reg_wei4, reg_wei4, reg_wei_stride);
        cmp(reg_tail, 1);
        b(LE, Lq_after_fill);
        ld1(VReg(0).h[1], ptr(reg_src));
        ld1(VReg(1).h[1], ptr(reg_wei));
        ld1(VReg(2).h[1], ptr(reg_wei2));
        ld1(VReg(3).h[1], ptr(reg_wei3));
        ld1(VReg(4).h[1], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        add(reg_wei3, reg_wei3, reg_wei_stride);
        add(reg_wei4, reg_wei4, reg_wei_stride);
        cmp(reg_tail, 2);
        b(LE, Lq_after_fill);
        ld1(VReg(0).h[2], ptr(reg_src));
        ld1(VReg(1).h[2], ptr(reg_wei));
        ld1(VReg(2).h[2], ptr(reg_wei2));
        ld1(VReg(3).h[2], ptr(reg_wei3));
        ld1(VReg(4).h[2], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        add(reg_wei3, reg_wei3, reg_wei_stride);
        add(reg_wei4, reg_wei4, reg_wei_stride);
        cmp(reg_tail, 3);
        b(LE, Lq_after_fill);
        ld1(VReg(0).h[3], ptr(reg_src));
        ld1(VReg(1).h[3], ptr(reg_wei));
        ld1(VReg(2).h[3], ptr(reg_wei2));
        ld1(VReg(3).h[3], ptr(reg_wei3));
        ld1(VReg(4).h[3], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        add(reg_wei3, reg_wei3, reg_wei_stride);
        add(reg_wei4, reg_wei4, reg_wei_stride);
        cmp(reg_tail, 4);
        b(LE, Lq_after_fill);
        ld1(VReg(0).h[4], ptr(reg_src));
        ld1(VReg(1).h[4], ptr(reg_wei));
        ld1(VReg(2).h[4], ptr(reg_wei2));
        ld1(VReg(3).h[4], ptr(reg_wei3));
        ld1(VReg(4).h[4], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        add(reg_wei3, reg_wei3, reg_wei_stride);
        add(reg_wei4, reg_wei4, reg_wei_stride);
        cmp(reg_tail, 5);
        b(LE, Lq_after_fill);
        ld1(VReg(0).h[5], ptr(reg_src));
        ld1(VReg(1).h[5], ptr(reg_wei));
        ld1(VReg(2).h[5], ptr(reg_wei2));
        ld1(VReg(3).h[5], ptr(reg_wei3));
        ld1(VReg(4).h[5], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        add(reg_wei3, reg_wei3, reg_wei_stride);
        add(reg_wei4, reg_wei4, reg_wei_stride);
        cmp(reg_tail, 6);
        b(LE, Lq_after_fill);
        ld1(VReg(0).h[6], ptr(reg_src));
        ld1(VReg(1).h[6], ptr(reg_wei));
        ld1(VReg(2).h[6], ptr(reg_wei2));
        ld1(VReg(3).h[6], ptr(reg_wei3));
        ld1(VReg(4).h[6], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        add(reg_wei3, reg_wei3, reg_wei_stride);
        add(reg_wei4, reg_wei4, reg_wei_stride);
        cmp(reg_tail, 7);
        b(LE, Lq_after_fill);
        ld1(VReg(0).h[7], ptr(reg_src));
        ld1(VReg(1).h[7], ptr(reg_wei));
        ld1(VReg(2).h[7], ptr(reg_wei2));
        ld1(VReg(3).h[7], ptr(reg_wei3));
        ld1(VReg(4).h[7], ptr(reg_wei4));
        L(Lq_after_fill);
        fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal(VReg4S(21), VReg4H(0), VReg4H(2));
        fmlal2(VReg4S(21), VReg4H(0), VReg4H(2));
        fmlal(VReg4S(22), VReg4H(0), VReg4H(3));
        fmlal2(VReg4S(22), VReg4H(0), VReg4H(3));
        fmlal(VReg4S(23), VReg4H(0), VReg4H(4));
        fmlal2(VReg4S(23), VReg4H(0), VReg4H(4));

        add(q_src_base, q_src_base, reg_src_dx);
        add(q_wei_base, q_wei_base, reg_wei_dx);
        add(q_wei2_base, q_wei2_base, reg_wei_dx);
        add(q_wei3_base, q_wei3_base, reg_wei_dx);
        add(q_wei4_base, q_wei4_base, reg_wei_dx);
        subs(reg_kw_cnt, reg_kw_cnt, 1);
        b(GT, Lq_kx_loop);

        // reduce/store 4 accumulators
        faddp(VReg4S(20), VReg4S(20), VReg4S(20));
        faddp(VReg2S(20), VReg2S(20), VReg2S(20));
        faddp(VReg4S(21), VReg4S(21), VReg4S(21));
        faddp(VReg2S(21), VReg2S(21), VReg2S(21));
        faddp(VReg4S(22), VReg4S(22), VReg4S(22));
        faddp(VReg2S(22), VReg2S(22), VReg2S(22));
        faddp(VReg4S(23), VReg4S(23), VReg4S(23));
        faddp(VReg2S(23), VReg2S(23), VReg2S(23));
        add(q_src_base, q_src_base, reg_src_dy);
        add(q_wei_base, q_wei_base, reg_wei_dy);
        add(q_wei2_base, q_wei2_base, reg_wei_dy);
        add(q_wei3_base, q_wei3_base, reg_wei_dy);
        add(q_wei4_base, q_wei4_base, reg_wei_dy);
        subs(q_kh_work, q_kh_work, 1);
        b(GT, Lq_ky_loop);

        // After ky loop: reduce/store
        ldr(SReg(0), ptr(reg_acc));
        fadd(SReg(0), SReg(0), SReg(20));
        str(SReg(0), ptr(reg_acc));
        ldr(SReg(1), ptr(reg_acc2));
        fadd(SReg(1), SReg(1), SReg(21));
        str(SReg(1), ptr(reg_acc2));
        ldr(SReg(2), ptr(reg_acc3));
        fadd(SReg(2), SReg(2), SReg(22));
        str(SReg(2), ptr(reg_acc3));
        ldr(SReg(3), ptr(reg_acc4));
        fadd(SReg(3), SReg(3), SReg(23));
        str(SReg(3), ptr(reg_acc4));
        b(Lend_all);

        // Not-packed fallback for quad: do lane-wise loads for 4 outputs
        L(Lq_np);
        // Zero v20..v23
        eor(VReg16B(20), VReg16B(20), VReg16B(20));
        eor(VReg16B(21), VReg16B(21), VReg16B(21));
        eor(VReg16B(22), VReg16B(22), VReg16B(22));
        eor(VReg16B(23), VReg16B(23), VReg16B(23));
        // single block (not looped here) — rely on host collapsing work
        // lanes 0..7
        ld1(VReg(0).h[0], ptr(reg_src));
        ld1(VReg(1).h[0], ptr(reg_wei));
        ld1(VReg(2).h[0], ptr(reg_wei2));
        ld1(VReg(3).h[0], ptr(reg_wei3));
        ld1(VReg(4).h[0], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        add(reg_wei3, reg_wei3, reg_wei_stride);
        add(reg_wei4, reg_wei4, reg_wei_stride);
        ld1(VReg(0).h[1], ptr(reg_src));
        ld1(VReg(1).h[1], ptr(reg_wei));
        ld1(VReg(2).h[1], ptr(reg_wei2));
        ld1(VReg(3).h[1], ptr(reg_wei3));
        ld1(VReg(4).h[1], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        add(reg_wei3, reg_wei3, reg_wei_stride);
        add(reg_wei4, reg_wei4, reg_wei_stride);
        ld1(VReg(0).h[2], ptr(reg_src));
        ld1(VReg(1).h[2], ptr(reg_wei));
        ld1(VReg(2).h[2], ptr(reg_wei2));
        ld1(VReg(3).h[2], ptr(reg_wei3));
        ld1(VReg(4).h[2], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        add(reg_wei3, reg_wei3, reg_wei_stride);
        add(reg_wei4, reg_wei4, reg_wei_stride);
        ld1(VReg(0).h[3], ptr(reg_src));
        ld1(VReg(1).h[3], ptr(reg_wei));
        ld1(VReg(2).h[3], ptr(reg_wei2));
        ld1(VReg(3).h[3], ptr(reg_wei3));
        ld1(VReg(4).h[3], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        add(reg_wei3, reg_wei3, reg_wei_stride);
        add(reg_wei4, reg_wei4, reg_wei_stride);
        ld1(VReg(0).h[4], ptr(reg_src));
        ld1(VReg(1).h[4], ptr(reg_wei));
        ld1(VReg(2).h[4], ptr(reg_wei2));
        ld1(VReg(3).h[4], ptr(reg_wei3));
        ld1(VReg(4).h[4], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        add(reg_wei3, reg_wei3, reg_wei_stride);
        add(reg_wei4, reg_wei4, reg_wei_stride);
        ld1(VReg(0).h[5], ptr(reg_src));
        ld1(VReg(1).h[5], ptr(reg_wei));
        ld1(VReg(2).h[5], ptr(reg_wei2));
        ld1(VReg(3).h[5], ptr(reg_wei3));
        ld1(VReg(4).h[5], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        add(reg_wei3, reg_wei3, reg_wei_stride);
        add(reg_wei4, reg_wei4, reg_wei_stride);
        ld1(VReg(0).h[6], ptr(reg_src));
        ld1(VReg(1).h[6], ptr(reg_wei));
        ld1(VReg(2).h[6], ptr(reg_wei2));
        ld1(VReg(3).h[6], ptr(reg_wei3));
        ld1(VReg(4).h[6], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        add(reg_wei3, reg_wei3, reg_wei_stride);
        add(reg_wei4, reg_wei4, reg_wei_stride);
        ld1(VReg(0).h[7], ptr(reg_src));
        ld1(VReg(1).h[7], ptr(reg_wei));
        ld1(VReg(2).h[7], ptr(reg_wei2));
        ld1(VReg(3).h[7], ptr(reg_wei3));
        ld1(VReg(4).h[7], ptr(reg_wei4));
        fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal(VReg4S(21), VReg4H(0), VReg4H(2));
        fmlal2(VReg4S(21), VReg4H(0), VReg4H(2));
        fmlal(VReg4S(22), VReg4H(0), VReg4H(3));
        fmlal2(VReg4S(22), VReg4H(0), VReg4H(3));
        fmlal(VReg4S(23), VReg4H(0), VReg4H(4));
        fmlal2(VReg4S(23), VReg4H(0), VReg4H(4));
        // reduce/store
        faddp(VReg4S(20), VReg4S(20), VReg4S(20));
        faddp(VReg2S(20), VReg2S(20), VReg2S(20));
        faddp(VReg4S(21), VReg4S(21), VReg4S(21));
        faddp(VReg2S(21), VReg2S(21), VReg2S(21));
        faddp(VReg4S(22), VReg4S(22), VReg4S(22));
        faddp(VReg2S(22), VReg2S(22), VReg2S(22));
        faddp(VReg4S(23), VReg4S(23), VReg4S(23));
        faddp(VReg2S(23), VReg2S(23), VReg2S(23));
        ldr(SReg(0), ptr(reg_acc));
        fadd(SReg(0), SReg(0), SReg(20));
        str(SReg(0), ptr(reg_acc));
        ldr(SReg(1), ptr(reg_acc2));
        fadd(SReg(1), SReg(1), SReg(21));
        str(SReg(1), ptr(reg_acc2));
        ldr(SReg(2), ptr(reg_acc3));
        fadd(SReg(2), SReg(2), SReg(22));
        str(SReg(2), ptr(reg_acc3));
        ldr(SReg(3), ptr(reg_acc4));
        fadd(SReg(3), SReg(3), SReg(23));
        str(SReg(3), ptr(reg_acc4));
        b(Lend_all);
    }
    // ---------------- Dual-OC path ----------------
    {
        // Zero FP32 accumulators v20.4s and v21.4s
        eor(VReg16B(20), VReg16B(20), VReg16B(20));
        eor(VReg16B(21), VReg16B(21), VReg16B(21));
        Label Lnp_d, Ld_after_fill, Ld_after_loop, Ld_kx_loop, Ld_after_kx;
        // If not packed, fallback to non-packed path below (no kw-loop optimization)
        cmp(reg_wei_stride, 2);
        b(NE, Lnp_d);
        // Save initial repeats and bases
        const XReg reg_reps_init = x15;
        const XReg reg_src_base = x16;
        const XReg reg_wei_base = x17;
        const XReg reg_wei2_base = x18;
        mov(reg_reps_init, reg_reps);
        mov(reg_src_base, reg_src);
        mov(reg_wei_base, reg_wei);
        mov(reg_wei2_base, reg_wei2);
        const XReg reg_kw_init = x28;
        const XReg reg_kh_work = x29;
        mov(reg_kw_init, reg_kw_cnt);
        mov(reg_kh_work, reg_kh_cnt);
        // ky-loop wrapper around kx-loop (packed fast path)
        Label Ld_ky_loop;
        L(Ld_ky_loop);
        // Reset per-ky state and restore kw counter
        mov(reg_reps, reg_reps_init);
        mov(reg_src, reg_src_base);
        mov(reg_wei, reg_wei_base);
        mov(reg_wei2, reg_wei2_base);
        mov(reg_kw_cnt, reg_kw_init);
        // kw-loop (only if kw_cnt > 0); if kw_cnt==0 -> process a single position
        L(Ld_kx_loop);
        // Main loop over full 8-lane channel blocks
        Label Ld_loop;
        L(Ld_loop);
        cmp(reg_reps, 0);
        b(LE, Ld_after_loop);
        // Load 8 src half lanes with strides between channels
        ld1(VReg(0).h[0], ptr(reg_src));
        add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[1], ptr(reg_src));
        add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[2], ptr(reg_src));
        add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[3], ptr(reg_src));
        add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[4], ptr(reg_src));
        add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[5], ptr(reg_src));
        add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[6], ptr(reg_src));
        add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[7], ptr(reg_src));
        // Load 8 half weights for oc0 and oc1 as vectors
        ld1(VReg8H(1), ptr(reg_wei));
        ld1(VReg8H(2), ptr(reg_wei2));
        // MAC for oc0 → v20, oc1 → v21
        fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal(VReg4S(21), VReg4H(0), VReg4H(2));
        fmlal2(VReg4S(21), VReg4H(0), VReg4H(2));
        // Advance pointers to next block (src already advanced by 8*src_stride)
        add(reg_wei, reg_wei, reg_wei_blk_stride);
        add(reg_wei2, reg_wei2, reg_wei_blk_stride);
        sub(reg_reps, reg_reps, 1);
        b(Ld_loop);
        // Tail processing (<=8)
        L(Ld_after_loop);
        eor(VReg16B(0), VReg16B(0), VReg16B(0));
        eor(VReg16B(1), VReg16B(1), VReg16B(1));
        eor(VReg16B(2), VReg16B(2), VReg16B(2));
        cmp(reg_tail, 0);
        b(LE, Ld_after_fill);
        ld1(VReg(0).h[0], ptr(reg_src));
        ld1(VReg(1).h[0], ptr(reg_wei));
        ld1(VReg(2).h[0], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        cmp(reg_tail, 1);
        b(LE, Ld_after_fill);
        ld1(VReg(0).h[1], ptr(reg_src));
        ld1(VReg(1).h[1], ptr(reg_wei));
        ld1(VReg(2).h[1], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        cmp(reg_tail, 2);
        b(LE, Ld_after_fill);
        ld1(VReg(0).h[2], ptr(reg_src));
        ld1(VReg(1).h[2], ptr(reg_wei));
        ld1(VReg(2).h[2], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        cmp(reg_tail, 3);
        b(LE, Ld_after_fill);
        ld1(VReg(0).h[3], ptr(reg_src));
        ld1(VReg(1).h[3], ptr(reg_wei));
        ld1(VReg(2).h[3], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        cmp(reg_tail, 4);
        b(LE, Ld_after_fill);
        ld1(VReg(0).h[4], ptr(reg_src));
        ld1(VReg(1).h[4], ptr(reg_wei));
        ld1(VReg(2).h[4], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        cmp(reg_tail, 5);
        b(LE, Ld_after_fill);
        ld1(VReg(0).h[5], ptr(reg_src));
        ld1(VReg(1).h[5], ptr(reg_wei));
        ld1(VReg(2).h[5], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        cmp(reg_tail, 6);
        b(LE, Ld_after_fill);
        ld1(VReg(0).h[6], ptr(reg_src));
        ld1(VReg(1).h[6], ptr(reg_wei));
        ld1(VReg(2).h[6], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        cmp(reg_tail, 7);
        b(LE, Ld_after_fill);
        ld1(VReg(0).h[7], ptr(reg_src));
        ld1(VReg(1).h[7], ptr(reg_wei));
        ld1(VReg(2).h[7], ptr(reg_wei2));
        L(Ld_after_fill);
        fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal(VReg4S(21), VReg4H(0), VReg4H(2));
        fmlal2(VReg4S(21), VReg4H(0), VReg4H(2));
        // Advance base pointers for next kx
        add(reg_src_base, reg_src_base, reg_src_dx);
        add(reg_wei_base, reg_wei_base, reg_wei_dx);
        add(reg_wei2_base, reg_wei2_base, reg_wei_dx);
        // Decrement kw_cnt and loop
        subs(reg_kw_cnt, reg_kw_cnt, 1);
        b(GT, Ld_kx_loop);
        // After kx loop for one ky, advance to next ky if any
        add(reg_src_base, reg_src_base, reg_src_dy);
        add(reg_wei_base, reg_wei_base, reg_wei_dy);
        add(reg_wei2_base, reg_wei2_base, reg_wei_dy);
        subs(reg_kh_work, reg_kh_work, 1);
        b(GT, Ld_ky_loop);
        // Fallthrough to store
        b(Ld_after_kx);

        // Not-packed path: pairwise lane loads for both wei/wei2 (no kw-loop optimization)
        L(Lnp_d);
        ld1(VReg(0).h[0], ptr(reg_src));
        ld1(VReg(1).h[0], ptr(reg_wei));
        ld1(VReg(2).h[0], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        ld1(VReg(0).h[1], ptr(reg_src));
        ld1(VReg(1).h[1], ptr(reg_wei));
        ld1(VReg(2).h[1], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        ld1(VReg(0).h[2], ptr(reg_src));
        ld1(VReg(1).h[2], ptr(reg_wei));
        ld1(VReg(2).h[2], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        ld1(VReg(0).h[3], ptr(reg_src));
        ld1(VReg(1).h[3], ptr(reg_wei));
        ld1(VReg(2).h[3], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        ld1(VReg(0).h[4], ptr(reg_src));
        ld1(VReg(1).h[4], ptr(reg_wei));
        ld1(VReg(2).h[4], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        ld1(VReg(0).h[5], ptr(reg_src));
        ld1(VReg(1).h[5], ptr(reg_wei));
        ld1(VReg(2).h[5], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        ld1(VReg(0).h[6], ptr(reg_src));
        ld1(VReg(1).h[6], ptr(reg_wei));
        ld1(VReg(2).h[6], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        ld1(VReg(0).h[7], ptr(reg_src));
        ld1(VReg(1).h[7], ptr(reg_wei));
        ld1(VReg(2).h[7], ptr(reg_wei2));
        // Accumulate
        fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal(VReg4S(21), VReg4H(0), VReg4H(2));
        fmlal2(VReg4S(21), VReg4H(0), VReg4H(2));
        // Advance to next block
        add(reg_src, reg_src, reg_src_stride);
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        sub(reg_reps, reg_reps, 1);
        b(Ld_loop);

        // Reduce and store both accumulators
        L(Ld_after_kx);
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
        b(Lend_all);
    }

    // ---------------- Single-OC path (default) ----------------
    L(Lsingle);
    // Zero FP32 accumulator v20.4s
    eor(VReg16B(20), VReg16B(20), VReg16B(20));

    Label Lloop2, Lloop, Lafter_loop, Lafter_fill, Lnp1, Lend1, Lnp2, Lend2, Lnot_packed, Lkx_loop_s, Lafter_kx_s,
        Ls_loop, Ls_after_loop, Ls_after_fill;

    // Unrolled-by-2 loop over full 8-lane channel blocks (default single position)
    b(Lloop);
    L(Lloop2);
    cmp(reg_reps, 2);
    b(LT, Lloop);

    // First block (packed fast path if available)
    cmp(reg_wei_stride, 2);
    b(NE, Lnp1);
    // packed: src lanes (8), then vector-load weights
    ld1(VReg(0).h[0], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[1], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[2], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[3], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[4], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[5], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[6], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[7], ptr(reg_src));
    ld1(VReg8H(1), ptr(reg_wei));
    fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
    fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_blk_stride);
    b(Lend1);
    L(Lnp1);
    // not packed: pairwise lanes
    ld1(VReg(0).h[0], ptr(reg_src));
    ld1(VReg(1).h[0], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[1], ptr(reg_src));
    ld1(VReg(1).h[1], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[2], ptr(reg_src));
    ld1(VReg(1).h[2], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[3], ptr(reg_src));
    ld1(VReg(1).h[3], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[4], ptr(reg_src));
    ld1(VReg(1).h[4], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[5], ptr(reg_src));
    ld1(VReg(1).h[5], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[6], ptr(reg_src));
    ld1(VReg(1).h[6], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[7], ptr(reg_src));
    ld1(VReg(1).h[7], ptr(reg_wei));
    fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
    fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    L(Lend1);

    // Second block
    cmp(reg_wei_stride, 2);
    b(NE, Lnp2);
    ld1(VReg(0).h[0], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[1], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[2], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[3], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[4], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[5], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[6], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[7], ptr(reg_src));
    ld1(VReg8H(1), ptr(reg_wei));
    fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
    fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_blk_stride);
    b(Lend2);
    L(Lnp2);
    ld1(VReg(0).h[0], ptr(reg_src));
    ld1(VReg(1).h[0], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[1], ptr(reg_src));
    ld1(VReg(1).h[1], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[2], ptr(reg_src));
    ld1(VReg(1).h[2], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[3], ptr(reg_src));
    ld1(VReg(1).h[3], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[4], ptr(reg_src));
    ld1(VReg(1).h[4], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[5], ptr(reg_src));
    ld1(VReg(1).h[5], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[6], ptr(reg_src));
    ld1(VReg(1).h[6], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[7], ptr(reg_src));
    ld1(VReg(1).h[7], ptr(reg_wei));
    fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
    fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    L(Lend2);

    sub(reg_reps, reg_reps, 2);
    b(Lloop2);

    // Single-block loop for remaining one block
    L(Lloop);
    cmp(reg_reps, 0);
    b(EQ, Lafter_loop);

    // Prepare containers for src/wei half vectors in v0/v1 (fully overwritten by loads)

    // Choose packed-weight fast path if wei_stride == 2 bytes
    cmp(reg_wei_stride, 2);
    b(NE, Lnot_packed);

    // Packed path: fill src lanes, vector-load wei
    ld1(VReg(0).h[0], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[1], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[2], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[3], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[4], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[5], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[6], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).h[7], ptr(reg_src));
    // load 8 half wei as one vector
    ld1(VReg8H(1), ptr(reg_wei));
    // Multiply-accumulate (fp16 widen to fp32): lower and upper halves
    fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
    fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
    // Advance pointers to next block
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_blk_stride);
    sub(reg_reps, reg_reps, 1);
    b(Lloop);

    // Not-packed path: fill src/wei lanes pairwise and accumulate
    L(Lnot_packed);
    ld1(VReg(0).h[0], ptr(reg_src));
    ld1(VReg(1).h[0], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[1], ptr(reg_src));
    ld1(VReg(1).h[1], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[2], ptr(reg_src));
    ld1(VReg(1).h[2], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[3], ptr(reg_src));
    ld1(VReg(1).h[3], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[4], ptr(reg_src));
    ld1(VReg(1).h[4], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[5], ptr(reg_src));
    ld1(VReg(1).h[5], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[6], ptr(reg_src));
    ld1(VReg(1).h[6], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(0).h[7], ptr(reg_src));
    ld1(VReg(1).h[7], ptr(reg_wei));
    // Widening MAC for both halves
    fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
    fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
    // Advance pointers to next block (src already advanced by 8*src_stride)
    add(reg_wei, reg_wei, reg_wei_stride);
    sub(reg_reps, reg_reps, 1);
    b(Lloop);

    L(Lafter_loop);

    // Tail processing (<= 8)
    // Prepare containers for src/wei half vectors in v0/v1
    eor(VReg16B(0), VReg16B(0), VReg16B(0));
    eor(VReg16B(1), VReg16B(1), VReg16B(1));
    // lane 0
    cmp(reg_tail, 0);
    b(LE, Lafter_fill);
    ld1(VReg(0).h[0], ptr(reg_src));
    ld1(VReg(1).h[0], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    // lane 1
    cmp(reg_tail, 1);
    b(LE, Lafter_fill);
    ld1(VReg(0).h[1], ptr(reg_src));
    ld1(VReg(1).h[1], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    // lane 2
    cmp(reg_tail, 2);
    b(LE, Lafter_fill);
    ld1(VReg(0).h[2], ptr(reg_src));
    ld1(VReg(1).h[2], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    // lane 3
    cmp(reg_tail, 3);
    b(LE, Lafter_fill);
    ld1(VReg(0).h[3], ptr(reg_src));
    ld1(VReg(1).h[3], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    // lane 4
    cmp(reg_tail, 4);
    b(LE, Lafter_fill);
    ld1(VReg(0).h[4], ptr(reg_src));
    ld1(VReg(1).h[4], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    // lane 5
    cmp(reg_tail, 5);
    b(LE, Lafter_fill);
    ld1(VReg(0).h[5], ptr(reg_src));
    ld1(VReg(1).h[5], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    // lane 6
    cmp(reg_tail, 6);
    b(LE, Lafter_fill);
    ld1(VReg(0).h[6], ptr(reg_src));
    ld1(VReg(1).h[6], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    // lane 7
    cmp(reg_tail, 7);
    b(LE, Lafter_fill);
    ld1(VReg(0).h[7], ptr(reg_src));
    ld1(VReg(1).h[7], ptr(reg_wei));

    L(Lafter_fill);
    // Accumulate tail using widening fp16 MAC
    fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
    fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));

    // Horizontal add v20
    faddp(VReg4S(20), VReg4S(20), VReg4S(20));
    faddp(VReg2S(20), VReg2S(20), VReg2S(20));

    // Load *acc, add, store back
    ldr(SReg(0), ptr(reg_acc));
    fadd(SReg(0), SReg(0), SReg(20));
    str(SReg(0), ptr(reg_acc));

    L(Lend_all);
    // Epilog: restore callee-saved
    ldp(XReg(29), XReg(30), post_ptr(sp, 16));
    ldp(XReg(27), XReg(28), post_ptr(sp, 16));
    ldp(XReg(25), XReg(26), post_ptr(sp, 16));
    ldp(XReg(23), XReg(24), post_ptr(sp, 16));
    ldp(XReg(21), XReg(22), post_ptr(sp, 16));
    ldp(XReg(19), XReg(20), post_ptr(sp, 16));
    ret();
}

void JitConv3DKernelF16::generate() {
    // Keep body small for clang-tidy readability-function-size
    gen_minimal_kernel();
    gen_optimized_kernel();
}


[[maybe_unused]] static inline auto ptr_f16(const MemoryPtr& mem) -> const uint16_t* {
    return reinterpret_cast<const uint16_t*>(mem->getData());
}
[[maybe_unused]] static inline auto ptr_f16(MemoryPtr& mem) -> uint16_t* {
    return reinterpret_cast<uint16_t*>(mem->getData());
}

JitConv3DExecutor::JitConv3DExecutor(const ConvAttrs& attrs,
                                     const MemoryArgs& memory,
                                     const ExecutorContext::CPtr& context)
    : m_attrs(attrs),
      m_memory(memory) {
    (void)context;
    m_threadsNum = static_cast<size_t>(parallel_get_max_threads());
    // Decide precision from src tensor
    ov::element::Type sp = ov::element::dynamic;
    auto it = memory.find(ARG_SRC);
    if (it != memory.end() && it->second && it->second->getDescPtr()) {
        sp = it->second->getDescPtr()->getPrecision();
    }
    m_is_fp32 = (sp == ov::element::f32);
    if (m_is_fp32) {
        m_ip_kernel_f32 = std::make_unique<JitConv3DKernelF32>();
        m_ip_kernel_f32->create_ker();
    } else {
        // default to fp16
        m_ip_kernel = std::make_unique<JitConv3DKernelF16>();
        m_ip_kernel->create_ker();
    }

    // Extract optional PReLU post-op (per-tensor or per-channel), but keep disabled by default.
    for (const auto& po : m_attrs.postOps) {
        if (const auto* const ss = std::any_cast<ScaleShiftPostOp>(&po)) {
            if (ss->type() == ScaleShiftPostOp::Type::prelu) {
                m_has_prelu = true;
                m_prelu_slopes = ss->scales();
                break;
            }
        }
    }

    // Early weight packing (only if shapes are static). Kept inside executor per policy.
    prepare_weights_early(m_memory);
}

bool JitConv3DExecutor::supports(const ConvConfig& cfg) {
    // Require 5D NCDHW, FP16 or FP32 src/wei/dst, group=1, no dilation, stride 1 or 2
    if (!cfg.descs.count(ARG_SRC) || !cfg.descs.count(ARG_WEI) || !cfg.descs.count(ARG_DST))
        return false;
    if (!cfg.descs.at(ARG_SRC) || !cfg.descs.at(ARG_WEI) || !cfg.descs.at(ARG_DST))
        return false;

    const auto& s = cfg.descs.at(ARG_SRC)->getShape();
    const auto& w = cfg.descs.at(ARG_WEI)->getShape();
    const auto& d = cfg.descs.at(ARG_DST)->getShape();
    if (s.getRank() != 5 || w.getRank() < 5 || d.getRank() != 5)
        return false;

    const auto sp = cfg.descs.at(ARG_SRC)->getPrecision();
    const auto wp = cfg.descs.at(ARG_WEI)->getPrecision();
    const auto dp = cfg.descs.at(ARG_DST)->getPrecision();
    const bool f16_ok = (sp == ov::element::f16 && wp == ov::element::f16 && dp == ov::element::f16);
    const bool f32_ok = (sp == ov::element::f32 && wp == ov::element::f32 && dp == ov::element::f32);
    if (!(f16_ok || f32_ok))
        return false;

    // group == 1: weights rank==5 (no groups)
    if (w.getRank() != 5)
        return false;

    // dilation == 0
    for (auto v : cfg.attrs.dilation) {
        if (v != 0)
            return false;
    }
    // stride in [1,2] if set
    for (auto v : cfg.attrs.stride) {
        if (!(v == 1 || v == 2))
            return false;
    }
    return true;
}

void JitConv3DExecutor::prepare_weights_early(const MemoryArgs& memory) {
    // Guard: only when shapes are static
    auto src_it = memory.find(ARG_SRC);
    auto wei_it = memory.find(ARG_WEI);
    if (src_it == memory.end() || wei_it == memory.end() || !src_it->second || !wei_it->second)
        return;
    const auto& s = src_it->second->getDescPtr()->getShape();
    const auto& w = wei_it->second->getDescPtr()->getShape();
    if (!s.isStatic() || !w.isStatic())
        return;
    if (m_is_fp32) {
        ensure_weights_packed_f32(memory);
    } else {
        ensure_weights_packed(memory);
    }
}

void JitConv3DExecutor::run_naive_fp16(const MemoryArgs& memory) {
    // NCDHW
    auto src = memory.at(ARG_SRC);
    auto wei = memory.at(ARG_WEI);
    auto dst = memory.at(ARG_DST);
    const auto& srcDims = src->getDescPtr()->getShape().getStaticDims();
    const auto& weiDims = wei->getDescPtr()->getShape().getStaticDims();
    const auto& dstDims = dst->getDescPtr()->getShape().getStaticDims();

    const size_t N = srcDims[0];
    const size_t C = srcDims[1];
    const size_t ID = srcDims[2], IH = srcDims[3], IW = srcDims[4];
    const size_t OC = weiDims[0];
    const size_t KD = weiDims[2], KH = weiDims[3], KW = weiDims[4];
    const size_t OD = dstDims[2], OH = dstDims[3], OW = dstDims[4];

    const size_t SD = m_attrs.stride.size() > 0 ? m_attrs.stride[0] : 1;
    const size_t SH = m_attrs.stride.size() > 1 ? m_attrs.stride[1] : 1;
    const size_t SW = m_attrs.stride.size() > 2 ? m_attrs.stride[2] : 1;

    const size_t PD0 = m_attrs.paddingL.size() > 0 ? static_cast<size_t>(m_attrs.paddingL[0]) : 0;
    const size_t PH0 = m_attrs.paddingL.size() > 1 ? static_cast<size_t>(m_attrs.paddingL[1]) : 0;
    const size_t PW0 = m_attrs.paddingL.size() > 2 ? static_cast<size_t>(m_attrs.paddingL[2]) : 0;

    const uint16_t* src_p = ptr_f16(src);
    uint16_t* dst_p = ptr_f16(dst);

    auto index_src = [&](size_t n, size_t c, size_t z, size_t y, size_t x) -> size_t {
        return (((n * C + c) * ID + z) * IH + y) * IW + x;
    };
    auto index_dst = [&](size_t n, size_t oc, size_t z, size_t y, size_t x) -> size_t {
        return (((n * OC + oc) * OD + z) * OH + y) * OW + x;
    };

    // Prepare packed weights once
    ensure_weights_packed(memory);

    auto worker = [&](size_t n, size_t oc_quad, size_t od) {
        const size_t oc0 = oc_quad * 4;
        const size_t oc1 = oc0 + 1;
        const size_t oc2 = oc0 + 2;
        const size_t oc3 = oc0 + 3;
        const bool has_oc1 = oc1 < OC;
        const bool has_oc2 = oc2 < OC;
        const bool has_oc3 = oc3 < OC;
        const int64_t iz0 = static_cast<int64_t>(od * SD) - static_cast<int64_t>(PD0);
        for (size_t oh = 0; oh < OH; ++oh) {
            const int64_t iy0 = static_cast<int64_t>(oh * SH) - static_cast<int64_t>(PH0);
            for (size_t ow = 0; ow < OW; ++ow) {
                const int64_t ix0 = static_cast<int64_t>(ow * SW) - static_cast<int64_t>(PW0);

                float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

                const size_t src_c_stride_elems = ID * IH * IW;

                if (SD == 1 && SH == 1 && SW == 1) {
                    const ptrdiff_t kz_lo = std::max<ptrdiff_t>(0, -iz0);
                    const ptrdiff_t kz_hi =
                        std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KD) - 1, static_cast<ptrdiff_t>(ID) - 1 - iz0);
                    const ptrdiff_t ky_lo = std::max<ptrdiff_t>(0, -iy0);
                    const ptrdiff_t ky_hi =
                        std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KH) - 1, static_cast<ptrdiff_t>(IH) - 1 - iy0);
                    const ptrdiff_t kx_lo = std::max<ptrdiff_t>(0, -ix0);
                    const ptrdiff_t kx_hi =
                        std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KW) - 1, static_cast<ptrdiff_t>(IW) - 1 - ix0);
                    if (kz_lo <= kz_hi && ky_lo <= ky_hi && kx_lo <= kx_hi) {
                        const size_t kw_count = static_cast<size_t>(kx_hi - kx_lo + 1);
                        for (ptrdiff_t kz = kz_lo; kz <= kz_hi; ++kz) {
                            const size_t iz = static_cast<size_t>(iz0 + kz);
                            // iy/ix for ky_lo/kx_lo not needed; use iy2/ix2 per ky below

                            // Loop over ky in host; kernel handles kx via kw_cnt (always packed)
                            for (ptrdiff_t ky = ky_lo; ky <= ky_hi; ++ky) {
                                const size_t iy2 = static_cast<size_t>(iy0 + ky);
                                const size_t ix2 = static_cast<size_t>(ix0 + kx_lo);
                                const size_t s_base2 = index_src(n, 0, iz, iy2, ix2);
                                auto run_pair = [&](float* acc, float* acc2, size_t base0, size_t base1) {
                                    jit_conv3d_call_args aa{};
                                    aa.src = src_p + s_base2;
                                    aa.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                    aa.src_blk_stride = aa.src_stride * 8;
                                    aa.acc = acc;
                                    aa.acc2 = acc2;
                                    aa.repeats = C / 8;
                                    aa.tail = C % 8;
                                    aa.kw_cnt = kw_count;
                                    aa.src_dx = sizeof(uint16_t);
                                    aa.wei = m_wei_packed.data() + base0;
                                    if (acc2) aa.wei2 = m_wei_packed.data() + base1;
                                    aa.wei_stride = sizeof(uint16_t);
                                    aa.wei_blk_stride = aa.wei_stride * 8;
                                    aa.wei_dx = m_padded_C * sizeof(uint16_t);
                                    (*m_ip_kernel)(&aa);
                                };
                                const size_t base0 = (((oc0 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) * KW + static_cast<size_t>(kx_lo)) * m_padded_C;
                                const size_t base1 = has_oc1 ? (((oc1 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) * KW + static_cast<size_t>(kx_lo)) * m_padded_C : 0;
                                run_pair(&acc0, has_oc1 ? &acc1 : nullptr, base0, base1);
                                if (has_oc2) {
                                    const size_t base2 = (((oc2 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) * KW + static_cast<size_t>(kx_lo)) * m_padded_C;
                                    const size_t base3 = has_oc3 ? (((oc3 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) * KW + static_cast<size_t>(kx_lo)) * m_padded_C : 0;
                                    run_pair(&acc2, has_oc3 ? &acc3 : nullptr, base2, base3);
                                }
                            }
                        }
                    }
                } else {
                    for (size_t kz = 0; kz < KD; ++kz) {
                        const int64_t iz = iz0 + static_cast<int64_t>(kz);
                        if (iz < 0 || iz >= static_cast<int64_t>(ID))
                            continue;
                        for (size_t ky = 0; ky < KH; ++ky) {
                            const int64_t iy = iy0 + static_cast<int64_t>(ky);
                            if (iy < 0 || iy >= static_cast<int64_t>(IH))
                                continue;
                            for (size_t kx = 0; kx < KW; ++kx) {
                                const int64_t ix = ix0 + static_cast<int64_t>(kx);
                                if (ix < 0 || ix >= static_cast<int64_t>(IW))
                                    continue;
                                const size_t s_base0 = index_src(n,
                                                                 0,
                                                                 static_cast<size_t>(iz),
                                                                 static_cast<size_t>(iy),
                                                                 static_cast<size_t>(ix));
                                auto run_pair2 = [&](float* acc, float* acc2, size_t base0, size_t base1) {
                                    jit_conv3d_call_args aa{};
                                    aa.src = src_p + s_base0;
                                    aa.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                    aa.src_blk_stride = aa.src_stride * 8;
                                    aa.acc = acc;
                                    aa.acc2 = acc2;
                                    const size_t pack_base0 = base0;
                                    aa.wei = m_wei_packed.data() + pack_base0;
                                    aa.repeats = C / 8;
                                    aa.tail = C % 8;
                                    aa.wei_stride = sizeof(uint16_t);
                                    aa.wei_blk_stride = aa.wei_stride * 8;
                                    if (acc2) {
                                        const size_t pack_base1 = base1;
                                        aa.wei2 = m_wei_packed.data() + pack_base1;
                                    }
                                    (*m_ip_kernel)(&aa);
                                };
                                const size_t b0 = (((oc0 * KD + kz) * KH + ky) * KW + kx) * m_padded_C;
                                const size_t b1 = has_oc1 ? (((oc1 * KD + kz) * KH + ky) * KW + kx) * m_padded_C : 0;
                                run_pair2(&acc0, has_oc1 ? &acc1 : nullptr, b0, b1);
                                if (has_oc2) {
                                    const size_t b2 = (((oc2 * KD + kz) * KH + ky) * KW + kx) * m_padded_C;
                                    const size_t b3 = has_oc3 ? (((oc3 * KD + kz) * KH + ky) * KW + kx) * m_padded_C : 0;
                                    run_pair2(&acc2, has_oc3 ? &acc3 : nullptr, b2, b3);
                                }
                            }
                        }
                    }
                }
                // No optional post-ops in product mode

                dst_p[index_dst(n, oc0, od, oh, ow)] = ov::float16(acc0).to_bits();
                if (has_oc1)
                    dst_p[index_dst(n, oc1, od, oh, ow)] = ov::float16(acc1).to_bits();
                if (has_oc2)
                    dst_p[index_dst(n, oc2, od, oh, ow)] = ov::float16(acc2).to_bits();
                if (has_oc3)
                    dst_p[index_dst(n, oc3, od, oh, ow)] = ov::float16(acc3).to_bits();
            }
        }
    };

    ov::parallel_for3d(N, (OC + 3) / 4, OD, worker);
}

void JitConv3DExecutor::execute(const MemoryArgs& memory) {
    if (m_is_fp32)
        run_naive_fp32(memory);
    else
        run_naive_fp16(memory);
}

void JitConv3DExecutor::ensure_weights_packed_f32(const MemoryArgs& memory) {
    if (m_wei_packed_ready_f32)
        return;
    auto src = memory.at(ARG_SRC);
    auto wei = memory.at(ARG_WEI);
    const auto& srcDims = src->getDescPtr()->getShape().getStaticDims();
    const auto& weiDims = wei->getDescPtr()->getShape().getStaticDims();
    if (srcDims.size() != 5 || weiDims.size() != 5)
        return;
    const size_t C = srcDims[1];
    const size_t OC = weiDims[0];
    const size_t KD = weiDims[2], KH = weiDims[3], KW = weiDims[4];
    m_padded_C_f32 = (C + 3) / 4 * 4;
    const size_t total = OC * KD * KH * KW * m_padded_C_f32;
    m_wei_packed_f32.assign(total, 0.0f);
    const auto* wsrc = reinterpret_cast<const float*>(wei->getData());

    auto idx_wei_src = [&](size_t oc, size_t c, size_t kz, size_t ky, size_t kx) -> size_t {
        return ((((oc)*C + c) * KD + kz) * KH + ky) * KW + kx;
    };
    auto idx_wei_pack = [&](size_t oc, size_t c, size_t kz, size_t ky, size_t kx) -> size_t {
        const size_t base = (((oc * KD + kz) * KH + ky) * KW + kx) * m_padded_C_f32;
        const size_t blk = c / 4;
        const size_t lane = c % 4;
        return base + blk * 4 + lane;
    };

    for (size_t oc = 0; oc < OC; ++oc) {
        for (size_t kz = 0; kz < KD; ++kz) {
            for (size_t ky = 0; ky < KH; ++ky) {
                for (size_t kx = 0; kx < KW; ++kx) {
                    for (size_t c = 0; c < C; ++c) {
                        m_wei_packed_f32[idx_wei_pack(oc, c, kz, ky, kx)] = wsrc[idx_wei_src(oc, c, kz, ky, kx)];
                    }
                }
            }
        }
    }
    m_wei_packed_ready_f32 = true;
    // no global cache store
}

void JitConv3DExecutor::run_naive_fp32(const MemoryArgs& memory) {
    auto src = memory.at(ARG_SRC);
    auto wei = memory.at(ARG_WEI);
    auto dst = memory.at(ARG_DST);
    const auto& srcDims = src->getDescPtr()->getShape().getStaticDims();
    const auto& weiDims = wei->getDescPtr()->getShape().getStaticDims();
    const auto& dstDims = dst->getDescPtr()->getShape().getStaticDims();

    const size_t N = srcDims[0];
    const size_t C = srcDims[1];
    const size_t ID = srcDims[2], IH = srcDims[3], IW = srcDims[4];
    const size_t OC = weiDims[0];
    const size_t KD = weiDims[2], KH = weiDims[3], KW = weiDims[4];
    const size_t OD = dstDims[2], OH = dstDims[3], OW = dstDims[4];

    const size_t SD = m_attrs.stride.size() > 0 ? m_attrs.stride[0] : 1;
    const size_t SH = m_attrs.stride.size() > 1 ? m_attrs.stride[1] : 1;
    const size_t SW = m_attrs.stride.size() > 2 ? m_attrs.stride[2] : 1;

    const ptrdiff_t PD0 = m_attrs.paddingL.size() > 0 ? m_attrs.paddingL[0] : 0;
    const ptrdiff_t PH0 = m_attrs.paddingL.size() > 1 ? m_attrs.paddingL[1] : 0;
    const ptrdiff_t PW0 = m_attrs.paddingL.size() > 2 ? m_attrs.paddingL[2] : 0;

    const auto* src_p = reinterpret_cast<const float*>(src->getData());
    const auto* wei_p = reinterpret_cast<const float*>(wei->getData());
    auto* dst_p = reinterpret_cast<float*>(dst->getData());

    auto index_src = [&](size_t n, size_t c, size_t z, size_t y, size_t x) {
        return (((n * C + c) * ID + z) * IH + y) * IW + x;
    };
    auto index_dst = [&](size_t n, size_t c, size_t z, size_t y, size_t x) {
        return (((n * OC + c) * OD + z) * OH + y) * OW + x;
    };
    auto index_wei = [&](size_t oc, size_t c, size_t kz, size_t ky, size_t kx) {
        return ((((oc)*C + c) * KD + kz) * KH + ky) * KW + kx;
    };

    const size_t src_c_stride_elems = ID * IH * IW;  // elements between channels
    const size_t wei_c_stride_elems = KD * KH * KW;  // elements between weight channels

    ensure_weights_packed_f32(memory);

    ov::parallel_for2d(N, (OC + 3) / 4, [&](size_t n, size_t oc_quad) {
        const size_t oc0 = oc_quad * 4;
        const size_t oc1 = std::min(oc0 + 1, OC);
        const size_t oc2 = std::min(oc0 + 2, OC);
        const size_t oc3 = std::min(oc0 + 3, OC);
        const bool has_oc1 = oc1 < OC;
        const bool has_oc2 = oc2 < OC;
        const bool has_oc3 = oc3 < OC;

        for (size_t od = 0; od < OD; ++od) {
            const ptrdiff_t iz0 = static_cast<ptrdiff_t>(od) * static_cast<ptrdiff_t>(SD) - PD0;
            for (size_t oh = 0; oh < OH; ++oh) {
                const ptrdiff_t iy0 = static_cast<ptrdiff_t>(oh) * static_cast<ptrdiff_t>(SH) - PH0;
                for (size_t ow = 0; ow < OW; ++ow) {
                    const ptrdiff_t ix0 = static_cast<ptrdiff_t>(ow) * static_cast<ptrdiff_t>(SW) - PW0;

                    float acc0 = 0.0F, acc1 = 0.0F, acc2 = 0.0F, acc3 = 0.0F;

                    if (SD == 1 && SH == 1 && SW == 1) {
                        const ptrdiff_t kz_lo = std::max<ptrdiff_t>(0, -iz0);
                        const ptrdiff_t kz_hi =
                            std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KD) - 1, static_cast<ptrdiff_t>(ID) - 1 - iz0);
                        const ptrdiff_t ky_lo = std::max<ptrdiff_t>(0, -iy0);
                        const ptrdiff_t ky_hi =
                            std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KH) - 1, static_cast<ptrdiff_t>(IH) - 1 - iy0);
                        const ptrdiff_t kx_lo = std::max<ptrdiff_t>(0, -ix0);
                        const ptrdiff_t kx_hi =
                            std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KW) - 1, static_cast<ptrdiff_t>(IW) - 1 - ix0);
                        if (kz_lo <= kz_hi && ky_lo <= ky_hi && kx_lo <= kx_hi) {
                            const size_t kw_count = static_cast<size_t>(kx_hi - kx_lo + 1);
                            for (ptrdiff_t kz = kz_lo; kz <= kz_hi; ++kz) {
                                const size_t iz = static_cast<size_t>(iz0 + kz);
                                for (ptrdiff_t ky = ky_lo; ky <= ky_hi; ++ky) {
                                    const size_t iy = static_cast<size_t>(iy0 + ky);
                                    const size_t ix = static_cast<size_t>(ix0 + kx_lo);
                                    const size_t s_base = index_src(n, 0, iz, iy, ix);

                                    if (m_wei_packed_ready_f32) {
                                        // pair 0
                                        {
                                            jit_conv3d_f32_call_args a{};
                                            a.src = src_p + s_base;
                                            a.src_stride = src_c_stride_elems * sizeof(float);
                                            a.src_blk_stride = a.src_stride * 4;
                                            a.acc = &acc0;
                                            a.acc2 = has_oc1 ? &acc1 : nullptr;
                                            a.repeats = C / 4;
                                            a.tail = C % 4;
                                            a.kw_cnt = kw_count;
                                            a.src_dx = sizeof(float);
                                            const size_t base0 =
                                                (((oc0 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) *
                                                     KW +
                                                 static_cast<size_t>(kx_lo)) *
                                                m_padded_C_f32;
                                            a.wei = m_wei_packed_f32.data() + base0;
                                            if (has_oc1) {
                                                const size_t base1 = (((oc1 * KD + static_cast<size_t>(kz)) * KH +
                                                                       static_cast<size_t>(ky)) *
                                                                          KW +
                                                                      static_cast<size_t>(kx_lo)) *
                                                                     m_padded_C_f32;
                                                a.wei2 = m_wei_packed_f32.data() + base1;
                                            }
                                            a.wei_stride = sizeof(float);
                                            a.wei_blk_stride = a.wei_stride * 4;
                                            a.wei_dx = m_padded_C_f32 * sizeof(float);
                                            (*m_ip_kernel_f32)(&a);
                                        }
                                        // pair 1
                                        if (has_oc2) {
                                            jit_conv3d_f32_call_args a{};
                                            a.src = src_p + s_base;
                                            a.src_stride = src_c_stride_elems * sizeof(float);
                                            a.src_blk_stride = a.src_stride * 4;
                                            a.acc = &acc2;
                                            a.acc2 = has_oc3 ? &acc3 : nullptr;
                                            a.repeats = C / 4;
                                            a.tail = C % 4;
                                            a.kw_cnt = kw_count;
                                            a.src_dx = sizeof(float);
                                            const size_t base2 =
                                                (((oc2 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) *
                                                     KW +
                                                 static_cast<size_t>(kx_lo)) *
                                                m_padded_C_f32;
                                            a.wei = m_wei_packed_f32.data() + base2;
                                            if (has_oc3) {
                                                const size_t base3 = (((oc3 * KD + static_cast<size_t>(kz)) * KH +
                                                                       static_cast<size_t>(ky)) *
                                                                          KW +
                                                                      static_cast<size_t>(kx_lo)) *
                                                                     m_padded_C_f32;
                                                a.wei2 = m_wei_packed_f32.data() + base3;
                                            }
                                            a.wei_stride = sizeof(float);
                                            a.wei_blk_stride = a.wei_stride * 4;
                                            a.wei_dx = m_padded_C_f32 * sizeof(float);
                                            (*m_ip_kernel_f32)(&a);
                                        }
                                    } else {
                                        // generic path: non-packed weights
                                        const size_t w0 = index_wei(oc0,
                                                                    0,
                                                                    static_cast<size_t>(kz),
                                                                    static_cast<size_t>(ky),
                                                                    static_cast<size_t>(kx_lo));
                                        const size_t w1 = has_oc1 ? index_wei(oc1,
                                                                              0,
                                                                              static_cast<size_t>(kz),
                                                                              static_cast<size_t>(ky),
                                                                              static_cast<size_t>(kx_lo))
                                                                  : 0;
                                        jit_conv3d_f32_call_args a{};
                                        a.src = src_p + s_base;
                                        a.src_stride = src_c_stride_elems * sizeof(float);
                                        a.src_blk_stride = a.src_stride * 4;
                                        a.acc = &acc0;
                                        a.acc2 = has_oc1 ? &acc1 : nullptr;
                                        a.repeats = C / 4;
                                        a.tail = C % 4;
                                        a.kw_cnt = kw_count;
                                        a.src_dx = sizeof(float);
                                        a.wei = wei_p + w0;
                                        if (has_oc1)
                                            a.wei2 = wei_p + w1;
                                        a.wei_stride = wei_c_stride_elems * sizeof(float);
                                        a.wei_blk_stride = a.wei_stride * 4;
                                        a.wei_dx = sizeof(float);
                                        (*m_ip_kernel_f32)(&a);
                                        if (has_oc2) {
                                            const size_t w2 = index_wei(oc2, 0, static_cast<size_t>(kz), static_cast<size_t>(ky), static_cast<size_t>(kx_lo));
                                            const size_t w3 = has_oc3 ? index_wei(oc3, 0, static_cast<size_t>(kz), static_cast<size_t>(ky), static_cast<size_t>(kx_lo)) : 0;
                                            jit_conv3d_f32_call_args a2{a};
                                            a2.acc = &acc2;
                                            a2.acc2 = has_oc3 ? &acc3 : nullptr;
                                            a2.wei = wei_p + w2;
                                            if (has_oc3) a2.wei2 = wei_p + w3;
                                            (*m_ip_kernel_f32)(&a2);
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // generic spatial stride path (host loops over all taps)
                        for (size_t kz = 0; kz < KD; ++kz) {
                            const ptrdiff_t iz = iz0 + static_cast<ptrdiff_t>(kz);
                            if (iz < 0 || iz >= static_cast<ptrdiff_t>(ID))
                                continue;
                            for (size_t ky = 0; ky < KH; ++ky) {
                                const ptrdiff_t iy = iy0 + static_cast<ptrdiff_t>(ky);
                                if (iy < 0 || iy >= static_cast<ptrdiff_t>(IH))
                                    continue;
                                for (size_t kx = 0; kx < KW; ++kx) {
                                    const ptrdiff_t ix = ix0 + static_cast<ptrdiff_t>(kx);
                                    if (ix < 0 || ix >= static_cast<ptrdiff_t>(IW))
                                        continue;
                                    const size_t s_base = index_src(n,
                                                                    0,
                                                                    static_cast<size_t>(iz),
                                                                    static_cast<size_t>(iy),
                                                                    static_cast<size_t>(ix));
                                    auto run_pair_f32 = [&](float* acc, float* acc2, const float* w0, const float* w1) {
                                        jit_conv3d_f32_call_args a{};
                                        a.src = src_p + s_base;
                                        a.src_stride = src_c_stride_elems * sizeof(float);
                                        a.src_blk_stride = a.src_stride * 4;
                                        a.acc = acc;
                                        a.acc2 = acc2;
                                        a.repeats = C / 4;
                                        a.tail = C % 4;
                                        a.kw_cnt = 1;
                                        a.src_dx = 0;
                                        a.wei = w0;
                                        if (w1) a.wei2 = w1;
                                        a.wei_stride = sizeof(float);
                                        a.wei_blk_stride = a.wei_stride * 4;
                                        a.wei_dx = 0;
                                        (*m_ip_kernel_f32)(&a);
                                    };
                                    const size_t base0 = (((oc0 * KD + kz) * KH + ky) * KW + kx) * m_padded_C_f32;
                                    const size_t base1 = has_oc1 ? (((oc1 * KD + kz) * KH + ky) * KW + kx) * m_padded_C_f32 : 0;
                                    run_pair_f32(&acc0, has_oc1 ? &acc1 : nullptr,
                                                 m_wei_packed_f32.data() + base0,
                                                 has_oc1 ? m_wei_packed_f32.data() + base1 : nullptr);
                                    if (has_oc2) {
                                        const size_t base2 = (((oc2 * KD + kz) * KH + ky) * KW + kx) * m_padded_C_f32;
                                        const size_t base3 = has_oc3 ? (((oc3 * KD + kz) * KH + ky) * KW + kx) * m_padded_C_f32 : 0;
                                        run_pair_f32(&acc2, has_oc3 ? &acc3 : nullptr,
                                                     m_wei_packed_f32.data() + base2,
                                                     has_oc3 ? m_wei_packed_f32.data() + base3 : nullptr);
                                    }
                                }
                            }
                        }
                    }

                    // Store FP32 accumulators directly to FP32 destination
                    dst_p[index_dst(n, oc0, od, oh, ow)] = acc0;
                    if (has_oc1)
                        dst_p[index_dst(n, oc1, od, oh, ow)] = acc1;
                    if (has_oc2)
                        dst_p[index_dst(n, oc2, od, oh, ow)] = acc2;
                    if (has_oc3)
                        dst_p[index_dst(n, oc3, od, oh, ow)] = acc3;
                }
            }
        }
    });
}

}  // namespace ov::intel_cpu
void ov::intel_cpu::JitConv3DExecutor::ensure_weights_packed(const MemoryArgs& memory) {
    if (m_wei_packed_ready)
        return;
    auto src = memory.at(ARG_SRC);
    auto wei = memory.at(ARG_WEI);
    const auto& srcDims = src->getDescPtr()->getShape().getStaticDims();
    const auto& weiDims = wei->getDescPtr()->getShape().getStaticDims();
    if (srcDims.size() != 5 || weiDims.size() != 5)
        return;
    const size_t C = srcDims[1];
    const size_t OC = weiDims[0];
    const size_t KD = weiDims[2], KH = weiDims[3], KW = weiDims[4];
    m_padded_C = (C + 7) / 8 * 8;
    // Pack layout: [OC, KD, KH, KW, Ct]
    const size_t total = OC * KD * KH * KW * m_padded_C;
    m_wei_packed.assign(total, static_cast<uint16_t>(0));
    const auto* wsrc = reinterpret_cast<const uint16_t*>(wei->getData());

    auto idx_wei_src = [&](size_t oc, size_t c, size_t kz, size_t ky, size_t kx) -> size_t {
        return ((((oc)*C + c) * KD + kz) * KH + ky) * KW + kx;
    };
    auto idx_wei_pack = [&](size_t oc, size_t c, size_t kz, size_t ky, size_t kx) -> size_t {
        const size_t base = (((oc * KD + kz) * KH + ky) * KW + kx) * m_padded_C;
        const size_t blk = c / 8;
        const size_t lane = c % 8;
        return base + blk * 8 + lane;
    };

    for (size_t oc = 0; oc < OC; ++oc) {
        for (size_t kz = 0; kz < KD; ++kz) {
            for (size_t ky = 0; ky < KH; ++ky) {
                for (size_t kx = 0; kx < KW; ++kx) {
                    for (size_t c = 0; c < C; ++c) {
                        m_wei_packed[idx_wei_pack(oc, c, kz, ky, kx)] = wsrc[idx_wei_src(oc, c, kz, ky, kx)];
                    }
                }
            }
        }
    }
    m_wei_packed_ready = true;
    // no global cache store
}
