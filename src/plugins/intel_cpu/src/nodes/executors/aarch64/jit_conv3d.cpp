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
#include <iostream>
#include "openvino/runtime/system_conf.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include <arm_neon.h>
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
    const bool use_fhm = m_use_fhm_;
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
    const XReg reg_src_blk_stride = x11; // size_t src_blk_stride (bytes)

    ldr(reg_src, ptr(reg_args, 0));
    ldr(reg_wei, ptr(reg_args, 8));
    ldr(reg_wei2, ptr(reg_args, 16));
    ldr(reg_reps, ptr(reg_args, 40));
    ldr(reg_tail, ptr(reg_args, 48));
    ldr(reg_src_stride, ptr(reg_args, 56));
    ldr(reg_wei_stride, ptr(reg_args, 64));
    ldr(reg_src_blk_stride, ptr(reg_args, 72));
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

    // helpers to emit identical load patterns without runtime cost
    auto emit_src8 = [&](const XReg& src, const XReg& src_stride) {
        Label Ls_np, Ls_done;
        cmp(src_stride, 2);
        b(NE, Ls_np);
        ld1(VReg8H(0), ptr(src));
        add(src, src, reg_src_blk_stride);
        b(Ls_done);
        L(Ls_np);
        ld1(VReg(0).h[0], ptr(src)); add(src, src, src_stride);
        ld1(VReg(0).h[1], ptr(src)); add(src, src, src_stride);
        ld1(VReg(0).h[2], ptr(src)); add(src, src, src_stride);
        ld1(VReg(0).h[3], ptr(src)); add(src, src, src_stride);
        ld1(VReg(0).h[4], ptr(src)); add(src, src, src_stride);
        ld1(VReg(0).h[5], ptr(src)); add(src, src, src_stride);
        ld1(VReg(0).h[6], ptr(src)); add(src, src, src_stride);
        ld1(VReg(0).h[7], ptr(src));
        L(Ls_done);
    };
    auto emit_wei8_pair = [&](const XReg& wei, const XReg& wei2, const XReg& wei_stride, const XReg& wei_blk_stride) {
        Label Lw_np, Lw_done;
        cmp(wei_stride, 2);
        b(NE, Lw_np);
        ld1(VReg8H(1), ptr(wei));
        ld1(VReg8H(2), ptr(wei2));
        add(wei, wei, wei_blk_stride);
        add(wei2, wei2, wei_blk_stride);
        b(Lw_done);
        L(Lw_np);
        ld1(VReg(1).h[0], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(2).h[0], ptr(wei2)); add(wei2, wei2, wei_stride);
        ld1(VReg(1).h[1], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(2).h[1], ptr(wei2)); add(wei2, wei2, wei_stride);
        ld1(VReg(1).h[2], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(2).h[2], ptr(wei2)); add(wei2, wei2, wei_stride);
        ld1(VReg(1).h[3], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(2).h[3], ptr(wei2)); add(wei2, wei2, wei_stride);
        ld1(VReg(1).h[4], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(2).h[4], ptr(wei2)); add(wei2, wei2, wei_stride);
        ld1(VReg(1).h[5], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(2).h[5], ptr(wei2)); add(wei2, wei2, wei_stride);
        ld1(VReg(1).h[6], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(2).h[6], ptr(wei2)); add(wei2, wei2, wei_stride);
        ld1(VReg(1).h[7], ptr(wei));
        ld1(VReg(2).h[7], ptr(wei2));
        L(Lw_done);
    };
    auto emit_wei8_single_blk = [&](const XReg& wei, const XReg& wei_stride, const XReg& wei_blk_stride) {
        Label Lw_np, Lw_done;
        cmp(wei_stride, 2);
        b(NE, Lw_np);
        ld1(VReg8H(1), ptr(wei));
        add(wei, wei, wei_blk_stride);
        b(Lw_done);
        L(Lw_np);
        ld1(VReg(1).h[0], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(1).h[1], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(1).h[2], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(1).h[3], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(1).h[4], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(1).h[5], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(1).h[6], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(1).h[7], ptr(wei));
        L(Lw_done);
    };
    auto emit_wei8_single16 = [&](const XReg& wei, const XReg& wei_stride) {
        Label Lw_np, Lw_done;
        cmp(wei_stride, 2);
        b(NE, Lw_np);
        ld1(VReg8H(1), ptr(wei));
        add(wei, wei, 16);
        b(Lw_done);
        L(Lw_np);
        ld1(VReg(1).h[0], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(1).h[1], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(1).h[2], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(1).h[3], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(1).h[4], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(1).h[5], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(1).h[6], ptr(wei)); add(wei, wei, wei_stride);
        ld1(VReg(1).h[7], ptr(wei));
        L(Lw_done);
    };
    L(Lkx_d);
    ldr(reg_reps, ptr(reg_args, 40));
    mov(reg_src, q_src_base);
    mov(reg_wei, q_wei_base);
    mov(reg_wei2, q_wei2_base);
    Label Lrep_d_kx;
    L(Lrep_d_kx);
    cmp(reg_reps, 0);
    b(EQ, Ltail_prep_d_kx);
    emit_src8(reg_src, reg_src_stride);
    emit_wei8_pair(reg_wei, reg_wei2, reg_wei_stride, reg_wei_blk_stride2);
    if (use_fhm) {
        fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal(VReg4S(21), VReg4H(0), VReg4H(2));
        fmlal2(VReg4S(21), VReg4H(0), VReg4H(2));
    } else {
        fcvtl(VReg4S(24), VReg4H(0));   // src low -> s
        fcvtl(VReg4S(25), VReg4H(1));   // w1 low -> s
        fmla(VReg4S(20), VReg4S(24), VReg4S(25));
        fcvtl2(VReg4S(26), VReg8H(0));  // src high -> s
        fcvtl2(VReg4S(27), VReg8H(1));  // w1 high -> s
        fmla(VReg4S(20), VReg4S(26), VReg4S(27));
        fcvtl(VReg4S(28), VReg4H(2));   // w2 low -> s
        fmla(VReg4S(21), VReg4S(24), VReg4S(28));
        fcvtl2(VReg4S(29), VReg8H(2));  // w2 high -> s
        fmla(VReg4S(21), VReg4S(26), VReg4S(29));
    }
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
    if (use_fhm) {
        fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal(VReg4S(21), VReg4H(0), VReg4H(2));
        fmlal2(VReg4S(21), VReg4H(0), VReg4H(2));
    } else {
        fcvtl(VReg4S(24), VReg4H(0));
        fcvtl(VReg4S(25), VReg4H(1));
        fmla(VReg4S(20), VReg4S(24), VReg4S(25));
        fcvtl2(VReg4S(26), VReg8H(0));
        fcvtl2(VReg4S(27), VReg8H(1));
        fmla(VReg4S(20), VReg4S(26), VReg4S(27));
        fcvtl(VReg4S(28), VReg4H(2));
        fmla(VReg4S(21), VReg4S(24), VReg4S(28));
        fcvtl2(VReg4S(29), VReg8H(2));
        fmla(VReg4S(21), VReg4S(26), VReg4S(29));
    }
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
    emit_src8(reg_src, reg_src_stride);
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
    if (use_fhm) {
        fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal(VReg4S(21), VReg4H(0), VReg4H(2));
        fmlal2(VReg4S(21), VReg4H(0), VReg4H(2));
    } else {
        fcvtl(VReg4S(24), VReg4H(0));
        fcvtl(VReg4S(25), VReg4H(1));
        fmla(VReg4S(20), VReg4S(24), VReg4S(25));
        fcvtl2(VReg4S(26), VReg8H(0));
        fcvtl2(VReg4S(27), VReg8H(1));
        fmla(VReg4S(20), VReg4S(26), VReg4S(27));
        fcvtl(VReg4S(28), VReg4H(2));
        fmla(VReg4S(21), VReg4S(24), VReg4S(28));
        fcvtl2(VReg4S(29), VReg8H(2));
        fmla(VReg4S(21), VReg4S(26), VReg4S(29));
    }
    sub(reg_reps, reg_reps, 1);
    b(Lrep_d);

    L(Ltail_prep_d);
    eor(VReg16B(0), VReg16B(0), VReg16B(0));
    eor(VReg16B(1), VReg16B(1), VReg16B(1));
    eor(VReg16B(2), VReg16B(2), VReg16B(2));
    // lanes 0..7 guarded by tail
    {
        Label Ltail_done_d;
        cmp(reg_tail, 0); b(LE, Ltail_done_d);
        ld1(VReg(0).h[0], ptr(reg_src)); ld1(VReg(1).h[0], ptr(reg_wei)); ld1(VReg(2).h[0], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride); add(reg_wei2, reg_wei2, reg_wei_stride);
        cmp(reg_tail, 1); b(LE, Ltail_done_d);
        ld1(VReg(0).h[1], ptr(reg_src)); ld1(VReg(1).h[1], ptr(reg_wei)); ld1(VReg(2).h[1], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride); add(reg_wei2, reg_wei2, reg_wei_stride);
        cmp(reg_tail, 2); b(LE, Ltail_done_d);
        ld1(VReg(0).h[2], ptr(reg_src)); ld1(VReg(1).h[2], ptr(reg_wei)); ld1(VReg(2).h[2], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride); add(reg_wei2, reg_wei2, reg_wei_stride);
        cmp(reg_tail, 3); b(LE, Ltail_done_d);
        ld1(VReg(0).h[3], ptr(reg_src)); ld1(VReg(1).h[3], ptr(reg_wei)); ld1(VReg(2).h[3], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride); add(reg_wei2, reg_wei2, reg_wei_stride);
        cmp(reg_tail, 4); b(LE, Ltail_done_d);
        ld1(VReg(0).h[4], ptr(reg_src)); ld1(VReg(1).h[4], ptr(reg_wei)); ld1(VReg(2).h[4], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride); add(reg_wei2, reg_wei2, reg_wei_stride);
        cmp(reg_tail, 5); b(LE, Ltail_done_d);
        ld1(VReg(0).h[5], ptr(reg_src)); ld1(VReg(1).h[5], ptr(reg_wei)); ld1(VReg(2).h[5], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride); add(reg_wei2, reg_wei2, reg_wei_stride);
        cmp(reg_tail, 6); b(LE, Ltail_done_d);
        ld1(VReg(0).h[6], ptr(reg_src)); ld1(VReg(1).h[6], ptr(reg_wei)); ld1(VReg(2).h[6], ptr(reg_wei2));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride); add(reg_wei2, reg_wei2, reg_wei_stride);
        cmp(reg_tail, 7); b(LE, Ltail_done_d);
        ld1(VReg(0).h[7], ptr(reg_src)); ld1(VReg(1).h[7], ptr(reg_wei)); ld1(VReg(2).h[7], ptr(reg_wei2));
        L(Ltail_done_d);
    }
    if (use_fhm) {
        fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal(VReg4S(21), VReg4H(0), VReg4H(2));
        fmlal2(VReg4S(21), VReg4H(0), VReg4H(2));
    } else {
        fcvtl(VReg4S(24), VReg4H(0));
        fcvtl(VReg4S(25), VReg4H(1));
        fmla(VReg4S(20), VReg4S(24), VReg4S(25));
        fcvtl2(VReg4S(26), VReg8H(0));
        fcvtl2(VReg4S(27), VReg8H(1));
        fmla(VReg4S(20), VReg4S(26), VReg4S(27));
        fcvtl(VReg4S(28), VReg4H(2));
        fmla(VReg4S(21), VReg4S(24), VReg4S(28));
        fcvtl2(VReg4S(29), VReg8H(2));
        fmla(VReg4S(21), VReg4S(26), VReg4S(29));
    }
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
    emit_wei8_single_blk(reg_wei, reg_wei_stride, s_wei_blk_stride2);
    if (use_fhm) {
        fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
    } else {
        fcvtl(VReg4S(24), VReg4H(0));
        fcvtl(VReg4S(25), VReg4H(1));
        fmla(VReg4S(20), VReg4S(24), VReg4S(25));
        fcvtl2(VReg4S(26), VReg8H(0));
        fcvtl2(VReg4S(27), VReg8H(1));
        fmla(VReg4S(20), VReg4S(26), VReg4S(27));
    }
    sub(reg_reps, reg_reps, 1);
    b(Lrep_s_kx);
    L(Ltail_prep_s_kx);
    eor(VReg16B(0), VReg16B(0), VReg16B(0));
    eor(VReg16B(1), VReg16B(1), VReg16B(1));
    {
        Label Ltail_done_s_kx;
        cmp(reg_tail, 0); b(LE, Ltail_done_s_kx);
        ld1(VReg(0).h[0], ptr(reg_src)); ld1(VReg(1).h[0], ptr(reg_wei));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride);
        cmp(reg_tail, 1); b(LE, Ltail_done_s_kx);
        ld1(VReg(0).h[1], ptr(reg_src)); ld1(VReg(1).h[1], ptr(reg_wei));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride);
        cmp(reg_tail, 2); b(LE, Ltail_done_s_kx);
        ld1(VReg(0).h[2], ptr(reg_src)); ld1(VReg(1).h[2], ptr(reg_wei));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride);
        cmp(reg_tail, 3); b(LE, Ltail_done_s_kx);
        ld1(VReg(0).h[3], ptr(reg_src)); ld1(VReg(1).h[3], ptr(reg_wei));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride);
        cmp(reg_tail, 4); b(LE, Ltail_done_s_kx);
        ld1(VReg(0).h[4], ptr(reg_src)); ld1(VReg(1).h[4], ptr(reg_wei));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride);
        cmp(reg_tail, 5); b(LE, Ltail_done_s_kx);
        ld1(VReg(0).h[5], ptr(reg_src)); ld1(VReg(1).h[5], ptr(reg_wei));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride);
        cmp(reg_tail, 6); b(LE, Ltail_done_s_kx);
        ld1(VReg(0).h[6], ptr(reg_src)); ld1(VReg(1).h[6], ptr(reg_wei));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride);
        cmp(reg_tail, 7); b(LE, Ltail_done_s_kx);
        ld1(VReg(0).h[7], ptr(reg_src)); ld1(VReg(1).h[7], ptr(reg_wei));
        L(Ltail_done_s_kx);
    }
    if (use_fhm) {
        fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
        fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
    } else {
        fcvtl(VReg4S(24), VReg4H(0));
        fcvtl(VReg4S(25), VReg4H(1));
        fmla(VReg4S(20), VReg4S(24), VReg4S(25));
        fcvtl2(VReg4S(26), VReg8H(0));
        fcvtl2(VReg4S(27), VReg8H(1));
        fmla(VReg4S(20), VReg4S(26), VReg4S(27));
    }
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
    emit_src8(reg_src, reg_src_stride);
    emit_wei8_single16(reg_wei, reg_wei_stride);
    fmlal(VReg4S(20), VReg4H(0), VReg4H(1));
    fmlal2(VReg4S(20), VReg4H(0), VReg4H(1));
    sub(reg_reps, reg_reps, 1);
    b(Lrep_s);

    // Tail (single)
    L(Ltail_prep_s);
    eor(VReg16B(0), VReg16B(0), VReg16B(0));
    eor(VReg16B(1), VReg16B(1), VReg16B(1));
    {
        Label Ltail_done_s;
        cmp(reg_tail, 0); b(LE, Ltail_done_s);
        ld1(VReg(0).h[0], ptr(reg_src)); ld1(VReg(1).h[0], ptr(reg_wei));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride);
        cmp(reg_tail, 1); b(LE, Ltail_done_s);
        ld1(VReg(0).h[1], ptr(reg_src)); ld1(VReg(1).h[1], ptr(reg_wei));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride);
        cmp(reg_tail, 2); b(LE, Ltail_done_s);
        ld1(VReg(0).h[2], ptr(reg_src)); ld1(VReg(1).h[2], ptr(reg_wei));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride);
        cmp(reg_tail, 3); b(LE, Ltail_done_s);
        ld1(VReg(0).h[3], ptr(reg_src)); ld1(VReg(1).h[3], ptr(reg_wei));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride);
        cmp(reg_tail, 4); b(LE, Ltail_done_s);
        ld1(VReg(0).h[4], ptr(reg_src)); ld1(VReg(1).h[4], ptr(reg_wei));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride);
        cmp(reg_tail, 5); b(LE, Ltail_done_s);
        ld1(VReg(0).h[5], ptr(reg_src)); ld1(VReg(1).h[5], ptr(reg_wei));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride);
        cmp(reg_tail, 6); b(LE, Ltail_done_s);
        ld1(VReg(0).h[6], ptr(reg_src)); ld1(VReg(1).h[6], ptr(reg_wei));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride);
        cmp(reg_tail, 7); b(LE, Ltail_done_s);
        ld1(VReg(0).h[7], ptr(reg_src)); ld1(VReg(1).h[7], ptr(reg_wei));
        L(Ltail_done_s);
    }
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
    const bool use_fhm = m_use_fhm_;
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
    Label Lq_entry, Lq_entry_nofhm;
    cbnz(reg_acc4, use_fhm ? Lq_entry : Lq_entry_nofhm);
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
        // Load 8 src lanes: vector fast path if src_stride==2 (packed Ct)
        Label Lsrc_np_q, Lsrc_done_q;
        cmp(reg_src_stride, 2);
        b(NE, Lsrc_np_q);
        ld1(VReg8H(0), ptr(reg_src));
        prfm(PLDL1KEEP, ptr(reg_src, 64));
        add(reg_src, reg_src, reg_src_blk_stride);
        b(Lsrc_done_q);
        L(Lsrc_np_q);
        ld1(VReg(0).h[0], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[1], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[2], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[3], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[4], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[5], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[6], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[7], ptr(reg_src));
        L(Lsrc_done_q);
        // Load 4 weight vectors
        ld1(VReg8H(1), ptr(reg_wei));
        ld1(VReg8H(2), ptr(reg_wei2));
        ld1(VReg8H(3), ptr(reg_wei3));
        ld1(VReg8H(4), ptr(reg_wei4));
        prfm(PLDL1KEEP, ptr(reg_wei, 64));
        prfm(PLDL1KEEP, ptr(reg_wei2, 64));
        prfm(PLDL1KEEP, ptr(reg_wei3, 64));
        prfm(PLDL1KEEP, ptr(reg_wei4, 64));
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

    // ---------------- Quad-OC path (no FHM: widen f16->f32 + fmla) ----------------
    L(Lq_entry_nofhm);
    {
        // Zero v20..v23 accumulators
        eor(VReg16B(20), VReg16B(20), VReg16B(20));
        eor(VReg16B(21), VReg16B(21), VReg16B(21));
        eor(VReg16B(22), VReg16B(22), VReg16B(22));
        eor(VReg16B(23), VReg16B(23), VReg16B(23));

        // Save repeats and bases for kw loop
        const XReg q_reps_init2 = x15;
        const XReg q_src_base2 = x16;
        const XReg q_wei_base2 = x17;
        const XReg q_wei2_base2 = x23;
        const XReg q_wei3_base2 = x24;
        const XReg q_wei4_base2 = x25;
        const XReg q_kw_init2 = x28;
        const XReg q_kh_work2 = x29;
        mov(q_reps_init2, reg_reps);
        mov(q_src_base2, reg_src);
        mov(q_wei_base2, reg_wei);
        mov(q_wei2_base2, reg_wei2);
        mov(q_wei3_base2, reg_wei3);
        mov(q_wei4_base2, reg_wei4);
        mov(q_kw_init2, reg_kw_cnt);
        mov(q_kh_work2, reg_kh_cnt);

        Label Lq2_ky_loop, Lq2_kx_loop, Lq2_loop, Lq2_after_loop, Lq2_after_fill, Lq2_after_kx, Lq2_tail_done;

        // ky loop
        L(Lq2_ky_loop);
        // kx loop entry
        L(Lq2_kx_loop);
        mov(reg_reps, q_reps_init2);
        mov(reg_src, q_src_base2);
        mov(reg_wei, q_wei_base2);
        mov(reg_wei2, q_wei2_base2);
        mov(reg_wei3, q_wei3_base2);
        mov(reg_wei4, q_wei4_base2);
        mov(reg_kw_cnt, q_kw_init2);

        // repeats over 8-lane C blocks
        L(Lq2_loop);
        cmp(reg_reps, 0);
        b(LE, Lq2_after_loop);
        // Load 8 src lanes: vector fast path if src_stride==2 (packed Ct)
        Label Lsrc_np_q2, Lsrc_done_q2;
        cmp(reg_src_stride, 2);
        b(NE, Lsrc_np_q2);
        ld1(VReg8H(0), ptr(reg_src));
        prfm(PLDL1KEEP, ptr(reg_src, 64));
        add(reg_src, reg_src, reg_src_blk_stride);
        b(Lsrc_done_q2);
        L(Lsrc_np_q2);
        ld1(VReg(0).h[0], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[1], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[2], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[3], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[4], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[5], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[6], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).h[7], ptr(reg_src));
        L(Lsrc_done_q2);
        // Load 4 weight vectors (packed, stride==2)
        ld1(VReg8H(1), ptr(reg_wei));
        ld1(VReg8H(2), ptr(reg_wei2));
        ld1(VReg8H(3), ptr(reg_wei3));
        ld1(VReg8H(4), ptr(reg_wei4));
        prfm(PLDL1KEEP, ptr(reg_wei, 64));
        prfm(PLDL1KEEP, ptr(reg_wei2, 64));
        prfm(PLDL1KEEP, ptr(reg_wei3, 64));
        prfm(PLDL1KEEP, ptr(reg_wei4, 64));
        // Widen src halves
        fcvtl(VReg4S(24), VReg4H(0));   // src low
        fcvtl2(VReg4S(26), VReg8H(0));  // src high
        // oc0/oc1: widen weights and fmla
        fcvtl(VReg4S(25), VReg4H(1));
        fcvtl2(VReg4S(27), VReg8H(1));
        fmla(VReg4S(20), VReg4S(24), VReg4S(25));
        fmla(VReg4S(20), VReg4S(26), VReg4S(27));
        fcvtl(VReg4S(28), VReg4H(2));
        fcvtl2(VReg4S(29), VReg8H(2));
        fmla(VReg4S(21), VReg4S(24), VReg4S(28));
        fmla(VReg4S(21), VReg4S(26), VReg4S(29));
        // oc2/oc3: reuse 25/27,28/29 regs
        fcvtl(VReg4S(25), VReg4H(3));
        fcvtl2(VReg4S(27), VReg8H(3));
        fmla(VReg4S(22), VReg4S(24), VReg4S(25));
        fmla(VReg4S(22), VReg4S(26), VReg4S(27));
        fcvtl(VReg4S(28), VReg4H(4));
        fcvtl2(VReg4S(29), VReg8H(4));
        fmla(VReg4S(23), VReg4S(24), VReg4S(28));
        fmla(VReg4S(23), VReg4S(26), VReg4S(29));
        // Advance packed weight block pointers
        add(reg_wei, reg_wei, reg_wei_blk_stride);
        add(reg_wei2, reg_wei2, reg_wei_blk_stride);
        add(reg_wei3, reg_wei3, reg_wei_blk_stride);
        add(reg_wei4, reg_wei4, reg_wei_blk_stride);
        sub(reg_reps, reg_reps, 1);
        b(Lq2_loop);

        // Tail <=8 C lanes
        L(Lq2_after_loop);
        eor(VReg16B(0), VReg16B(0), VReg16B(0));
        eor(VReg16B(1), VReg16B(1), VReg16B(1));
        eor(VReg16B(2), VReg16B(2), VReg16B(2));
        eor(VReg16B(3), VReg16B(3), VReg16B(3));
        eor(VReg16B(4), VReg16B(4), VReg16B(4));
        cmp(reg_tail, 0);
        b(LE, Lq2_after_fill);
        // fill tails for src + 4 weight vectors
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
        cmp(reg_tail, 1); b(LE, Lq2_after_fill);
        ld1(VReg(0).h[1], ptr(reg_src)); ld1(VReg(1).h[1], ptr(reg_wei)); ld1(VReg(2).h[1], ptr(reg_wei2)); ld1(VReg(3).h[1], ptr(reg_wei3)); ld1(VReg(4).h[1], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride); add(reg_wei2, reg_wei2, reg_wei_stride); add(reg_wei3, reg_wei3, reg_wei_stride); add(reg_wei4, reg_wei4, reg_wei_stride);
        cmp(reg_tail, 2); b(LE, Lq2_after_fill);
        ld1(VReg(0).h[2], ptr(reg_src)); ld1(VReg(1).h[2], ptr(reg_wei)); ld1(VReg(2).h[2], ptr(reg_wei2)); ld1(VReg(3).h[2], ptr(reg_wei3)); ld1(VReg(4).h[2], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride); add(reg_wei2, reg_wei2, reg_wei_stride); add(reg_wei3, reg_wei3, reg_wei_stride); add(reg_wei4, reg_wei4, reg_wei_stride);
        cmp(reg_tail, 3); b(LE, Lq2_after_fill);
        ld1(VReg(0).h[3], ptr(reg_src)); ld1(VReg(1).h[3], ptr(reg_wei)); ld1(VReg(2).h[3], ptr(reg_wei2)); ld1(VReg(3).h[3], ptr(reg_wei3)); ld1(VReg(4).h[3], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride); add(reg_wei2, reg_wei2, reg_wei_stride); add(reg_wei3, reg_wei3, reg_wei_stride); add(reg_wei4, reg_wei4, reg_wei_stride);
        cmp(reg_tail, 4); b(LE, Lq2_after_fill);
        ld1(VReg(0).h[4], ptr(reg_src)); ld1(VReg(1).h[4], ptr(reg_wei)); ld1(VReg(2).h[4], ptr(reg_wei2)); ld1(VReg(3).h[4], ptr(reg_wei3)); ld1(VReg(4).h[4], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride); add(reg_wei2, reg_wei2, reg_wei_stride); add(reg_wei3, reg_wei3, reg_wei_stride); add(reg_wei4, reg_wei4, reg_wei_stride);
        cmp(reg_tail, 5); b(LE, Lq2_after_fill);
        ld1(VReg(0).h[5], ptr(reg_src)); ld1(VReg(1).h[5], ptr(reg_wei)); ld1(VReg(2).h[5], ptr(reg_wei2)); ld1(VReg(3).h[5], ptr(reg_wei3)); ld1(VReg(4).h[5], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride); add(reg_wei2, reg_wei2, reg_wei_stride); add(reg_wei3, reg_wei3, reg_wei_stride); add(reg_wei4, reg_wei4, reg_wei_stride);
        cmp(reg_tail, 6); b(LE, Lq2_after_fill);
        ld1(VReg(0).h[6], ptr(reg_src)); ld1(VReg(1).h[6], ptr(reg_wei)); ld1(VReg(2).h[6], ptr(reg_wei2)); ld1(VReg(3).h[6], ptr(reg_wei3)); ld1(VReg(4).h[6], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride); add(reg_wei2, reg_wei2, reg_wei_stride); add(reg_wei3, reg_wei3, reg_wei_stride); add(reg_wei4, reg_wei4, reg_wei_stride);
        cmp(reg_tail, 7); b(LE, Lq2_after_fill);
        ld1(VReg(0).h[7], ptr(reg_src)); ld1(VReg(1).h[7], ptr(reg_wei)); ld1(VReg(2).h[7], ptr(reg_wei2)); ld1(VReg(3).h[7], ptr(reg_wei3)); ld1(VReg(4).h[7], ptr(reg_wei4));
        L(Lq2_after_fill);

        // Widen tails and accumulate
        fcvtl(VReg4S(24), VReg4H(0)); fcvtl2(VReg4S(26), VReg8H(0));
        fcvtl(VReg4S(25), VReg4H(1)); fcvtl2(VReg4S(27), VReg8H(1));
        fmla(VReg4S(20), VReg4S(24), VReg4S(25)); fmla(VReg4S(20), VReg4S(26), VReg4S(27));
        fcvtl(VReg4S(28), VReg4H(2)); fcvtl2(VReg4S(29), VReg8H(2));
        fmla(VReg4S(21), VReg4S(24), VReg4S(28)); fmla(VReg4S(21), VReg4S(26), VReg4S(29));
        fcvtl(VReg4S(25), VReg4H(3)); fcvtl2(VReg4S(27), VReg8H(3));
        fmla(VReg4S(22), VReg4S(24), VReg4S(25)); fmla(VReg4S(22), VReg4S(26), VReg4S(27));
        fcvtl(VReg4S(28), VReg4H(4)); fcvtl2(VReg4S(29), VReg8H(4));
        fmla(VReg4S(23), VReg4S(24), VReg4S(28)); fmla(VReg4S(23), VReg4S(26), VReg4S(29));

        // Horizontal add 4-wide floats -> 1 scalar and write back to 4 acc pointers
        faddp(VReg4S(20), VReg4S(20), VReg4S(20));
        faddp(VReg2S(20), VReg2S(20), VReg2S(20));
        faddp(VReg4S(21), VReg4S(21), VReg4S(21));
        faddp(VReg2S(21), VReg2S(21), VReg2S(21));
        faddp(VReg4S(22), VReg4S(22), VReg4S(22));
        faddp(VReg2S(22), VReg2S(22), VReg2S(22));
        faddp(VReg4S(23), VReg4S(23), VReg4S(23));
        faddp(VReg2S(23), VReg2S(23), VReg2S(23));
        ldr(SReg(0), ptr(reg_acc)); fadd(SReg(0), SReg(0), SReg(20)); str(SReg(0), ptr(reg_acc));
        ldr(SReg(1), ptr(reg_acc2)); fadd(SReg(1), SReg(1), SReg(21)); str(SReg(1), ptr(reg_acc2));
        ldr(SReg(2), ptr(reg_acc3)); fadd(SReg(2), SReg(2), SReg(22)); str(SReg(2), ptr(reg_acc3));
        ldr(SReg(3), ptr(reg_acc4)); fadd(SReg(3), SReg(3), SReg(23)); str(SReg(3), ptr(reg_acc4));

        // next kx
        sub(reg_kw_cnt, reg_kw_cnt, 1);
        add(q_src_base2, q_src_base2, reg_src_dx);
        add(q_wei_base2, q_wei_base2, reg_wei_dx);
        add(q_wei2_base2, q_wei2_base2, reg_wei_dx);
        add(q_wei3_base2, q_wei3_base2, reg_wei_dx);
        add(q_wei4_base2, q_wei4_base2, reg_wei_dx);
        cbnz(reg_kw_cnt, Lq2_kx_loop);

        // next ky
        sub(q_kh_work2, q_kh_work2, 1);
        add(q_src_base2, q_src_base2, reg_src_dy);
        add(q_wei_base2, q_wei_base2, reg_wei_dy);
        add(q_wei2_base2, q_wei2_base2, reg_wei_dy);
        add(q_wei3_base2, q_wei3_base2, reg_wei_dy);
        add(q_wei4_base2, q_wei4_base2, reg_wei_dy);
        cbnz(q_kh_work2, Lq2_ky_loop);

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
    // Avoid x18 (platform register). Use callee-saved x24/x25 for extra weight bases in quad path
    const XReg reg_wei3 = x24; // unused in dual-OC path
    const XReg reg_wei4 = x25; // unused in dual-OC path
    const XReg reg_reps = x4;
    const XReg reg_tail = x5;
    const XReg reg_src_stride = x6;
    const XReg reg_wei_stride = x7;
    const XReg reg_src_blk_stride = x8;
    const XReg reg_wei_blk_stride = x9;
    const XReg reg_acc = x10;
    const XReg reg_acc2 = x11;
    const XReg reg_acc3 = x20; // unused in dual-OC path
    const XReg reg_acc4 = x21; // unused in dual-OC path
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
    // Extra quad-OC fields will be loaded lazily inside quad path

    const XReg q_src_base = x15;
    const XReg q_wei_base = x16;
    const XReg q_wei2_base = x17;

    Label Lsingle, Ldone;
    Label Ldual_kx, Lkx_d, Ltail_prep_d_kx, Ltail_done_d_kx;
    Label Lsingle_kx, Lkx_s, Ltail_prep_s_kx, Ltail_done_s_kx;
    Label Lquad;

    // Prefer quad-OC when acc3 provided (handles 3 or 4 outputs)
    cbnz(reg_acc3, Lquad);
    cbz(reg_acc2, Lsingle);
    b(Ldual_kx);

    // Quad-OC path (updates up to 4 outputs)
    L(Lquad);
    {
        // Load extra pointers for quad path
        ldr(reg_wei3, ptr(reg_args, 112));
        ldr(reg_wei4, ptr(reg_args, 120));
        ldr(reg_acc3, ptr(reg_args, 128));
        ldr(reg_acc4, ptr(reg_args, 136));
        // Preserve frame + callee-saved registers used in this path
        stp(XReg(29), XReg(30), pre_ptr(sp, -16));
        stp(XReg(19), XReg(20), pre_ptr(sp, -16));
        stp(XReg(21), XReg(22), pre_ptr(sp, -16));
        stp(XReg(23), XReg(24), pre_ptr(sp, -16));
        stp(XReg(25), XReg(26), pre_ptr(sp, -16));

        // Working bases per kx step
        const XReg q_src = x15;
        const XReg q_w0 = x16;
        const XReg q_w1 = x17;
        const XReg q_w2 = x22;
        const XReg q_w3 = x23;

        mov(q_src, reg_src);
        mov(q_w0, reg_wei);
        mov(q_w1, reg_wei2);
        mov(q_w2, reg_wei3);
        mov(q_w3, reg_wei4);
        // If 3 outputs only, alias wei4 to wei3 to avoid null-deref loads
        Label Lw4_ok;
        cbnz(reg_acc4, Lw4_ok);
        mov(q_w3, q_w2);
        L(Lw4_ok);
        Label Lkx_q;
        cbnz(reg_kw_cnt, Lkx_q);
        mov(reg_kw_cnt, 1);

        // kx-loop entry
        L(Lkx_q);
        ldr(reg_reps, ptr(reg_args, 24));
        mov(reg_src, q_src);
        mov(reg_wei, q_w0);
        mov(reg_wei2, q_w1);
        mov(reg_wei3, q_w2);
        mov(reg_wei4, q_w3);

        // Zero accumulators v20..v23
        eor(VReg16B(20), VReg16B(20), VReg16B(20));
        eor(VReg16B(21), VReg16B(21), VReg16B(21));
        eor(VReg16B(22), VReg16B(22), VReg16B(22));
        eor(VReg16B(23), VReg16B(23), VReg16B(23));

        Label Lrep_q, Lrep_q_single, Lrep_q_unrolled, Ltail_prep_q, Ltail_done_q, Lafter_kx_q, Lskip_acc4;
        L(Lrep_q);
        cmp(reg_reps, 0);
        b(EQ, Ltail_prep_q);
        // Use 2x unroll on packed-weights path when reps >= 2
        // Conditions: have at least 2 repeats and weights are contiguous (wei_stride==4)
        cmp(reg_reps, 2);
        b(LT, Lrep_q_single);
        cmp(reg_wei_stride, 4);
        b(NE, Lrep_q_single);
        b(Lrep_q_unrolled);

        // -------- Unrolled iteration (2x) --------
        L(Lrep_q_unrolled);
        // First repeat
        // Load 4 src lanes
        ld1(VReg(0).s[0], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).s[1], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).s[2], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).s[3], ptr(reg_src));
        // Prefetch next src/wei
        prfm(PLDL1KEEP, ptr(reg_src, 96));
        // Load weights, prefer packed vectors
        ld1(VReg4S(1), ptr(reg_wei));
        ld1(VReg4S(2), ptr(reg_wei2));
        ld1(VReg4S(3), ptr(reg_wei3));
        ld1(VReg4S(4), ptr(reg_wei4));
        prfm(PLDL1KEEP, ptr(reg_wei, 64));
        prfm(PLDL1KEEP, ptr(reg_wei2, 64));
        prfm(PLDL1KEEP, ptr(reg_wei3, 64));
        prfm(PLDL1KEEP, ptr(reg_wei4, 64));
        add(reg_wei, reg_wei, reg_wei_blk_stride);
        add(reg_wei2, reg_wei2, reg_wei_blk_stride);
        add(reg_wei3, reg_wei3, reg_wei_blk_stride);
        add(reg_wei4, reg_wei4, reg_wei_blk_stride);
        // Advance src to next block start
        add(reg_src, reg_src, reg_src_stride);

        // Accumulate into 4 outputs
        fmla(VReg4S(20), VReg4S(0), VReg4S(1));
        fmla(VReg4S(21), VReg4S(0), VReg4S(2));
        fmla(VReg4S(22), VReg4S(0), VReg4S(3));
        fmla(VReg4S(23), VReg4S(0), VReg4S(4));

        // Second repeat (use v5 for src and v10..v13 for weights; avoid overlap with tail temps v24..v29)
        ld1(VReg(5).s[0], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(5).s[1], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(5).s[2], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(5).s[3], ptr(reg_src));
        prfm(PLDL1KEEP, ptr(reg_src, 96));
        ld1(VReg4S(10), ptr(reg_wei));
        ld1(VReg4S(11), ptr(reg_wei2));
        ld1(VReg4S(12), ptr(reg_wei3));
        ld1(VReg4S(13), ptr(reg_wei4));
        prfm(PLDL1KEEP, ptr(reg_wei, 64));
        prfm(PLDL1KEEP, ptr(reg_wei2, 64));
        prfm(PLDL1KEEP, ptr(reg_wei3, 64));
        prfm(PLDL1KEEP, ptr(reg_wei4, 64));
        add(reg_wei, reg_wei, reg_wei_blk_stride);
        add(reg_wei2, reg_wei2, reg_wei_blk_stride);
        add(reg_wei3, reg_wei3, reg_wei_blk_stride);
        add(reg_wei4, reg_wei4, reg_wei_blk_stride);
        add(reg_src, reg_src, reg_src_stride);
        fmla(VReg4S(20), VReg4S(5), VReg4S(10));
        fmla(VReg4S(21), VReg4S(5), VReg4S(11));
        fmla(VReg4S(22), VReg4S(5), VReg4S(12));
        fmla(VReg4S(23), VReg4S(5), VReg4S(13));

        sub(reg_reps, reg_reps, 2);
        b(Lrep_q);

        // -------- Single iteration --------
        L(Lrep_q_single);
        // Load 4 src lanes
        ld1(VReg(0).s[0], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).s[1], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).s[2], ptr(reg_src)); add(reg_src, reg_src, reg_src_stride);
        ld1(VReg(0).s[3], ptr(reg_src));
        // Prefetch next src/wei
        prfm(PLDL1KEEP, ptr(reg_src, 96));
        // Load weights, prefer packed vectors
        Label Lw_np_q, Lw_done_q;
        cmp(reg_wei_stride, 4);
        b(NE, Lw_np_q);
        ld1(VReg4S(1), ptr(reg_wei));
        ld1(VReg4S(2), ptr(reg_wei2));
        ld1(VReg4S(3), ptr(reg_wei3));
        ld1(VReg4S(4), ptr(reg_wei4));
        prfm(PLDL1KEEP, ptr(reg_wei, 64));
        prfm(PLDL1KEEP, ptr(reg_wei2, 64));
        prfm(PLDL1KEEP, ptr(reg_wei3, 64));
        prfm(PLDL1KEEP, ptr(reg_wei4, 64));
        add(reg_wei, reg_wei, reg_wei_blk_stride);
        add(reg_wei2, reg_wei2, reg_wei_blk_stride);
        add(reg_wei3, reg_wei3, reg_wei_blk_stride);
        add(reg_wei4, reg_wei4, reg_wei_blk_stride);
        b(Lw_done_q);
        L(Lw_np_q);
        ld1(VReg(1).s[0], ptr(reg_wei));  add(reg_wei, reg_wei, reg_wei_stride);
        ld1(VReg(2).s[0], ptr(reg_wei2)); add(reg_wei2, reg_wei2, reg_wei_stride);
        ld1(VReg(3).s[0], ptr(reg_wei3)); add(reg_wei3, reg_wei3, reg_wei_stride);
        ld1(VReg(4).s[0], ptr(reg_wei4)); add(reg_wei4, reg_wei4, reg_wei_stride);
        ld1(VReg(1).s[1], ptr(reg_wei));  add(reg_wei, reg_wei, reg_wei_stride);
        ld1(VReg(2).s[1], ptr(reg_wei2)); add(reg_wei2, reg_wei2, reg_wei_stride);
        ld1(VReg(3).s[1], ptr(reg_wei3)); add(reg_wei3, reg_wei3, reg_wei_stride);
        ld1(VReg(4).s[1], ptr(reg_wei4)); add(reg_wei4, reg_wei4, reg_wei_stride);
        ld1(VReg(1).s[2], ptr(reg_wei));  add(reg_wei, reg_wei, reg_wei_stride);
        ld1(VReg(2).s[2], ptr(reg_wei2)); add(reg_wei2, reg_wei2, reg_wei_stride);
        ld1(VReg(3).s[2], ptr(reg_wei3)); add(reg_wei3, reg_wei3, reg_wei_stride);
        ld1(VReg(4).s[2], ptr(reg_wei4)); add(reg_wei4, reg_wei4, reg_wei_stride);
        ld1(VReg(1).s[3], ptr(reg_wei));
        ld1(VReg(2).s[3], ptr(reg_wei2));
        ld1(VReg(3).s[3], ptr(reg_wei3));
        ld1(VReg(4).s[3], ptr(reg_wei4));
        add(reg_wei, reg_wei, reg_wei_stride);
        add(reg_wei2, reg_wei2, reg_wei_stride);
        add(reg_wei3, reg_wei3, reg_wei_stride);
        add(reg_wei4, reg_wei4, reg_wei_stride);
        L(Lw_done_q);
        // Advance src to next block start
        add(reg_src, reg_src, reg_src_stride);

        // Accumulate into 4 outputs
        fmla(VReg4S(20), VReg4S(0), VReg4S(1));
        fmla(VReg4S(21), VReg4S(0), VReg4S(2));
        fmla(VReg4S(22), VReg4S(0), VReg4S(3));
        fmla(VReg4S(23), VReg4S(0), VReg4S(4));

        sub(reg_reps, reg_reps, 1);
        b(Lrep_q);

        // Tail (<=4)
        L(Ltail_prep_q);
        eor(VReg16B(0), VReg16B(0), VReg16B(0));
        eor(VReg16B(1), VReg16B(1), VReg16B(1));
        eor(VReg16B(2), VReg16B(2), VReg16B(2));
        eor(VReg16B(3), VReg16B(3), VReg16B(3));
        eor(VReg16B(4), VReg16B(4), VReg16B(4));
        cmp(reg_tail, 0); b(LE, Ltail_done_q);
        ld1(VReg(0).s[0], ptr(reg_src)); ld1(VReg(1).s[0], ptr(reg_wei)); ld1(VReg(2).s[0], ptr(reg_wei2)); ld1(VReg(3).s[0], ptr(reg_wei3)); ld1(VReg(4).s[0], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride); add(reg_wei2, reg_wei2, reg_wei_stride); add(reg_wei3, reg_wei3, reg_wei_stride); add(reg_wei4, reg_wei4, reg_wei_stride);
        cmp(reg_tail, 1); b(LE, Ltail_done_q);
        ld1(VReg(0).s[1], ptr(reg_src)); ld1(VReg(1).s[1], ptr(reg_wei)); ld1(VReg(2).s[1], ptr(reg_wei2)); ld1(VReg(3).s[1], ptr(reg_wei3)); ld1(VReg(4).s[1], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride); add(reg_wei2, reg_wei2, reg_wei_stride); add(reg_wei3, reg_wei3, reg_wei_stride); add(reg_wei4, reg_wei4, reg_wei_stride);
        cmp(reg_tail, 2); b(LE, Ltail_done_q);
        ld1(VReg(0).s[2], ptr(reg_src)); ld1(VReg(1).s[2], ptr(reg_wei)); ld1(VReg(2).s[2], ptr(reg_wei2)); ld1(VReg(3).s[2], ptr(reg_wei3)); ld1(VReg(4).s[2], ptr(reg_wei4));
        add(reg_src, reg_src, reg_src_stride); add(reg_wei, reg_wei, reg_wei_stride); add(reg_wei2, reg_wei2, reg_wei_stride); add(reg_wei3, reg_wei3, reg_wei_stride); add(reg_wei4, reg_wei4, reg_wei_stride);
        cmp(reg_tail, 3); b(LE, Ltail_done_q);
        ld1(VReg(0).s[3], ptr(reg_src)); ld1(VReg(1).s[3], ptr(reg_wei)); ld1(VReg(2).s[3], ptr(reg_wei2)); ld1(VReg(3).s[3], ptr(reg_wei3)); ld1(VReg(4).s[3], ptr(reg_wei4));
        L(Ltail_done_q);
        fmla(VReg4S(20), VReg4S(0), VReg4S(1));
        fmla(VReg4S(21), VReg4S(0), VReg4S(2));
        fmla(VReg4S(22), VReg4S(0), VReg4S(3));
        fmla(VReg4S(23), VReg4S(0), VReg4S(4));

        // Reduce and store
        faddp(VReg4S(20), VReg4S(20), VReg4S(20)); faddp(VReg2S(20), VReg2S(20), VReg2S(20));
        faddp(VReg4S(21), VReg4S(21), VReg4S(21)); faddp(VReg2S(21), VReg2S(21), VReg2S(21));
        faddp(VReg4S(22), VReg4S(22), VReg4S(22)); faddp(VReg2S(22), VReg2S(22), VReg2S(22));
        faddp(VReg4S(23), VReg4S(23), VReg4S(23)); faddp(VReg2S(23), VReg2S(23), VReg2S(23));
        ldr(SReg(0), ptr(reg_acc));  fadd(SReg(0), SReg(0), SReg(20));  str(SReg(0), ptr(reg_acc));
        ldr(SReg(1), ptr(reg_acc2)); fadd(SReg(1), SReg(1), SReg(21)); str(SReg(1), ptr(reg_acc2));
        cbz(reg_acc3, Lafter_kx_q);
        ldr(SReg(2), ptr(reg_acc3)); fadd(SReg(2), SReg(2), SReg(22)); str(SReg(2), ptr(reg_acc3));
        L(Lafter_kx_q);
        cbz(reg_acc4, Lskip_acc4);
        ldr(SReg(3), ptr(reg_acc4)); fadd(SReg(3), SReg(3), SReg(23)); str(SReg(3), ptr(reg_acc4));
        L(Lskip_acc4);

        // Advance to next kx
        sub(reg_kw_cnt, reg_kw_cnt, 1);
        add(q_src, q_src, reg_src_dx);
        add(q_w0, q_w0, reg_wei_dx);
        add(q_w1, q_w1, reg_wei_dx);
        add(q_w2, q_w2, reg_wei_dx);
        add(q_w3, q_w3, reg_wei_dx);
        cbnz(reg_kw_cnt, Lkx_q);

        // Restore and finish
        ldp(XReg(25), XReg(26), post_ptr(sp, 16));
        ldp(XReg(23), XReg(24), post_ptr(sp, 16));
        ldp(XReg(21), XReg(22), post_ptr(sp, 16));
        ldp(XReg(19), XReg(20), post_ptr(sp, 16));
        ldp(XReg(29), XReg(30), post_ptr(sp, 16));
        b(Ldone);
    }

    L(Ldual_kx);
    eor(VReg16B(20), VReg16B(20), VReg16B(20));
    eor(VReg16B(21), VReg16B(21), VReg16B(21));
    // Dual-OC path only (v20,v21)

    mov(q_src_base, reg_src);
    mov(q_wei_base, reg_wei);
    mov(q_wei2_base, reg_wei2);
    // no extra base registers for wei3/wei4 to honor AArch64 callee-saved convention
    cbnz(reg_kw_cnt, Lkx_d);
    mov(reg_kw_cnt, 1);

    L(Lkx_d);
    ldr(reg_reps, ptr(reg_args, 24));
    mov(reg_src, q_src_base);
    mov(reg_wei, q_wei_base);
    mov(reg_wei2, q_wei2_base);
    // Dual-OC path: no extra weight bases

    Label Lrep_d;
    L(Lrep_d);
    cmp(reg_reps, 0);
    b(EQ, Ltail_prep_d_kx);
    // Try 2x unroll on packed-weights path when reps >= 2
    Label Lrep_d_single;
    cmp(reg_reps, 2);
    b(LT, Lrep_d_single);
    cmp(reg_wei_stride, 4);
    b(NE, Lrep_d_single);
    // ---- Unrolled iteration: process two repeats ----
    // First repeat: load src lanes into v0, vector-load weights v1/v2
    ld1(VReg(0).s[0], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[1], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[2], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[3], ptr(reg_src));
    prfm(PLDL1KEEP, ptr(reg_src, 64));
    ld1(VReg4S(1), ptr(reg_wei));
    ld1(VReg4S(2), ptr(reg_wei2));
    prfm(PLDL1KEEP, ptr(reg_wei, 64));
    prfm(PLDL1KEEP, ptr(reg_wei2, 64));
    add(reg_wei, reg_wei, reg_wei_blk_stride);
    add(reg_wei2, reg_wei2, reg_wei_blk_stride);
    add(reg_src, reg_src, reg_src_stride);
    fmla(VReg4S(20), VReg4S(0), VReg4S(1));
    fmla(VReg4S(21), VReg4S(0), VReg4S(2));

    // Second repeat: load src lanes into v5, vector-load weights v24/v25 (avoid v8..v15)
    ld1(VReg(5).s[0], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(5).s[1], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(5).s[2], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(5).s[3], ptr(reg_src));
    prfm(PLDL1KEEP, ptr(reg_src, 64));
    ld1(VReg4S(24), ptr(reg_wei));
    ld1(VReg4S(25), ptr(reg_wei2));
    prfm(PLDL1KEEP, ptr(reg_wei, 64));
    prfm(PLDL1KEEP, ptr(reg_wei2, 64));
    add(reg_wei, reg_wei, reg_wei_blk_stride);
    add(reg_wei2, reg_wei2, reg_wei_blk_stride);
    add(reg_src, reg_src, reg_src_stride);
    fmla(VReg4S(20), VReg4S(5), VReg4S(24));
    fmla(VReg4S(21), VReg4S(5), VReg4S(25));

    sub(reg_reps, reg_reps, 2);
    b(Lrep_d);
    // ---- Unrolled iteration: process two repeats ----
    // First repeat: load src lanes into v0, vector-load weights v1/v2
    ld1(VReg(0).s[0], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[1], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[2], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[3], ptr(reg_src));
    prfm(PLDL1KEEP, ptr(reg_src, 64));
    ld1(VReg4S(1), ptr(reg_wei));
    ld1(VReg4S(2), ptr(reg_wei2));
    prfm(PLDL1KEEP, ptr(reg_wei, 64));
    prfm(PLDL1KEEP, ptr(reg_wei2, 64));
    add(reg_wei, reg_wei, reg_wei_blk_stride);
    add(reg_wei2, reg_wei2, reg_wei_blk_stride);
    add(reg_src, reg_src, reg_src_stride);
    fmla(VReg4S(20), VReg4S(0), VReg4S(1));
    fmla(VReg4S(21), VReg4S(0), VReg4S(2));

    // Second repeat: load src lanes into v5, vector-load weights v24/v25 (avoid v8..v15)
    ld1(VReg(5).s[0], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(5).s[1], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(5).s[2], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(5).s[3], ptr(reg_src));
    prfm(PLDL1KEEP, ptr(reg_src, 64));
    ld1(VReg4S(24), ptr(reg_wei));
    ld1(VReg4S(25), ptr(reg_wei2));
    prfm(PLDL1KEEP, ptr(reg_wei, 64));
    prfm(PLDL1KEEP, ptr(reg_wei2, 64));
    add(reg_wei, reg_wei, reg_wei_blk_stride);
    add(reg_wei2, reg_wei2, reg_wei_blk_stride);
    add(reg_src, reg_src, reg_src_stride);
    fmla(VReg4S(20), VReg4S(5), VReg4S(24));
    fmla(VReg4S(21), VReg4S(5), VReg4S(25));

    sub(reg_reps, reg_reps, 2);
    b(Lrep_d);

    // ---- Single iteration fallback ----
    L(Lrep_d_single);
    ld1(VReg(0).s[0], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[1], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[2], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[3], ptr(reg_src));
    prfm(PLDL1KEEP, ptr(reg_src, 64));
    Label Lw_np_d, Lw_done_d;
    cmp(reg_wei_stride, 4);
    b(NE, Lw_np_d);
    ld1(VReg4S(1), ptr(reg_wei));
    ld1(VReg4S(2), ptr(reg_wei2));
    prfm(PLDL1KEEP, ptr(reg_wei, 64));
    prfm(PLDL1KEEP, ptr(reg_wei2, 64));
    add(reg_wei, reg_wei, reg_wei_blk_stride);
    add(reg_wei2, reg_wei2, reg_wei_blk_stride);
    // (quad-OC path disabled)
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
    // (quad-OC path disabled)
    L(Lw_done_d);
    add(reg_src, reg_src, reg_src_stride);
    fmla(VReg4S(20), VReg4S(0), VReg4S(1));
    fmla(VReg4S(21), VReg4S(0), VReg4S(2));
    // (quad-OC path disabled)
    sub(reg_reps, reg_reps, 1);
    b(Lrep_d);

    L(Ltail_prep_d_kx);
    eor(VReg16B(0), VReg16B(0), VReg16B(0));
    eor(VReg16B(1), VReg16B(1), VReg16B(1));
    eor(VReg16B(2), VReg16B(2), VReg16B(2));
    // clear only used vectors
    cmp(reg_tail, 0);
    b(LE, Ltail_done_d_kx);
    ld1(VReg(0).s[0], ptr(reg_src));
    ld1(VReg(1).s[0], ptr(reg_wei));
    ld1(VReg(2).s[0], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    // (quad-OC path disabled)
    cmp(reg_tail, 1);
    b(LE, Ltail_done_d_kx);
    ld1(VReg(0).s[1], ptr(reg_src));
    ld1(VReg(1).s[1], ptr(reg_wei));
    ld1(VReg(2).s[1], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    // (quad-OC path disabled)
    cmp(reg_tail, 2);
    b(LE, Ltail_done_d_kx);
    ld1(VReg(0).s[2], ptr(reg_src));
    ld1(VReg(1).s[2], ptr(reg_wei));
    ld1(VReg(2).s[2], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    // (quad-OC path disabled)
    cmp(reg_tail, 3);
    b(LE, Ltail_done_d_kx);
    ld1(VReg(0).s[3], ptr(reg_src));
    ld1(VReg(1).s[3], ptr(reg_wei));
    ld1(VReg(2).s[3], ptr(reg_wei2));
    L(Ltail_done_d_kx);
    fmla(VReg4S(20), VReg4S(0), VReg4S(1));
    fmla(VReg4S(21), VReg4S(0), VReg4S(2));
    // (quad-OC path disabled)
    sub(reg_kw_cnt, reg_kw_cnt, 1);
    add(q_src_base, q_src_base, reg_src_dx);
    add(q_wei_base, q_wei_base, reg_wei_dx);
    add(q_wei2_base, q_wei2_base, reg_wei_dx);
    // (quad-OC path disabled)
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
    // (quad-OC path disabled)
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
        // FP16 path: create JIT kernel and configure it based on HW caps
        // We never require ASIMDFHM: if it's absent, the generator uses f16->f32 widen + fmla
        // For safety on N1 (no FHM), keep the optimized FHM-only parts unused.
        m_ip_kernel = std::make_unique<JitConv3DKernelF16>();
        // Enforce unified FP16 behavior across ARM machines: do not rely on FHM
        m_ip_kernel->set_use_fhm(false);
        // Keep conservative kernel shape for stability; may be relaxed during tuning
        m_ip_kernel->set_force_single_kh(true);
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
    // Require 5D NCDHW, FP16 or FP32 src/wei/dst, group=1
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

    // Allow f16 even without FHM; executor has internal fallback when FHM is absent

    // group == 1: weights rank==5 (no groups)
    if (w.getRank() != 5)
        return false;

    // Do not restrict dilation/stride/layouts here; executor handles generic cases
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

                            // Loop over ky in host; kernel handles kx via kw_cnt
                            for (ptrdiff_t ky = ky_lo; ky <= ky_hi; ++ky) {
                                const size_t iy2 = static_cast<size_t>(iy0 + ky);
                                const size_t ix2 = static_cast<size_t>(ix0 + kx_lo);
                                const size_t s_base2 = index_src(n, 0, iz, iy2, ix2);
                                // Optional input prepack (stride=1): pack Ct=8 blocks contiguously for kw_count taps
                                const size_t repeats = C / 8;
                                const size_t tail_c = C % 8;
                                std::vector<uint16_t> packed_src;
                                if (kw_count > 1 && C >= 32) {
                                    packed_src.assign((repeats * 8 + 8) * kw_count, static_cast<uint16_t>(0));
                                    for (size_t kx = 0; kx < kw_count; ++kx) {
                                        const size_t base = kx * (repeats * 8 + 8);
                                        // full 8-lane channel blocks
                                        for (size_t rep = 0; rep < repeats; ++rep) {
                                            const size_t c0 = rep * 8;
                                            for (size_t lane = 0; lane < 8; ++lane) {
                                                const size_t c = c0 + lane;
                                                const size_t sidx = index_src(n, c, iz, iy2, ix2 + kx);
                                                packed_src[base + rep * 8 + lane] = src_p[sidx];
                                            }
                                        }
                                        // tail lanes (zeros for the rest)
                                        const size_t tail_off = base + repeats * 8;
                                        for (size_t lane = 0; lane < tail_c; ++lane) {
                                            const size_t c = repeats * 8 + lane;
                                            const size_t sidx = index_src(n, c, iz, iy2, ix2 + kx);
                                            packed_src[tail_off + lane] = src_p[sidx];
                                        }
                                    }
                                }

                                auto run_pair = [&](float* acc, float* acc2, size_t base0, size_t base1) {
                                    jit_conv3d_call_args aa{};
                                    if (!packed_src.empty()) {
                                        aa.src = packed_src.data();
                                        aa.src_stride = sizeof(uint16_t);
                                        aa.src_blk_stride = aa.src_stride * 8;  // 16 bytes
                                        aa.kw_cnt = kw_count;
                                        aa.src_dx = (repeats * 8 + 8) * sizeof(uint16_t);
                                    } else {
                                        aa.src = src_p + s_base2;
                                        aa.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                        aa.src_blk_stride = aa.src_stride * 8;
                                        aa.kw_cnt = kw_count;
                                        aa.src_dx = sizeof(uint16_t);
                                    }
                                    aa.acc = acc;
                                    aa.acc2 = acc2;
                                    aa.repeats = repeats;
                                    aa.tail = tail_c;
                                    aa.wei = m_wei_packed.data() + base0;
                                    if (acc2) aa.wei2 = m_wei_packed.data() + base1;
                                    aa.wei_stride = sizeof(uint16_t);
                                    aa.wei_blk_stride = aa.wei_stride * 8;
                                    aa.wei_dx = m_padded_C * sizeof(uint16_t);
                                    (*m_ip_kernel)(&aa);
                                };
                                auto run_quad = [&](float* acc, float* acc2, float* acc3, float* acc4,
                                                    size_t base0, size_t base1, size_t base2, size_t base3) {
                                    jit_conv3d_call_args aa{};
                                    if (!packed_src.empty()) {
                                        aa.src = packed_src.data();
                                        aa.src_stride = sizeof(uint16_t);
                                        aa.src_blk_stride = aa.src_stride * 8;
                                        aa.kw_cnt = kw_count;
                                        aa.src_dx = (repeats * 8 + 8) * sizeof(uint16_t);
                                    } else {
                                        aa.src = src_p + s_base2;
                                        aa.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                        aa.src_blk_stride = aa.src_stride * 8;
                                        aa.kw_cnt = kw_count;
                                        aa.src_dx = sizeof(uint16_t);
                                    }
                                    aa.acc = acc;
                                    aa.acc2 = acc2;
                                    aa.acc3 = acc3;
                                    aa.acc4 = acc4;
                                    aa.repeats = repeats;
                                    aa.tail = tail_c;
                                    aa.wei = m_wei_packed.data() + base0;
                                    aa.wei2 = m_wei_packed.data() + base1;
                                    aa.wei3 = m_wei_packed.data() + base2;
                                    aa.wei4 = m_wei_packed.data() + base3;
                                    aa.wei_stride = sizeof(uint16_t);
                                    aa.wei_blk_stride = aa.wei_stride * 8;
                                    aa.wei_dx = m_padded_C * sizeof(uint16_t);
                                    (*m_ip_kernel)(&aa);
                                };
                                const size_t base0 = (((oc0 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) * KW + static_cast<size_t>(kx_lo)) * m_padded_C;
                                const size_t base1 = has_oc1 ? (((oc1 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) * KW + static_cast<size_t>(kx_lo)) * m_padded_C : 0;
                                if (has_oc2) {
                                    const size_t base2 = (((oc2 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) * KW + static_cast<size_t>(kx_lo)) * m_padded_C;
                                    const size_t base3 = has_oc3 ? (((oc3 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) * KW + static_cast<size_t>(kx_lo)) * m_padded_C : 0;
                                    run_quad(&acc0, has_oc1 ? &acc1 : nullptr,
                                             &acc2, has_oc3 ? &acc3 : nullptr,
                                             base0, base1, base2, base3);
                                } else {
                                    run_pair(&acc0, has_oc1 ? &acc1 : nullptr, base0, base1);
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
                                auto run_quad2 = [&](float* acc, float* acc2, float* acc3, float* acc4,
                                                     size_t base0, size_t base1, size_t base2, size_t base3) {
                                    jit_conv3d_call_args aa{};
                                    aa.src = src_p + s_base0;
                                    aa.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                    aa.src_blk_stride = aa.src_stride * 8;
                                    aa.acc = acc;
                                    aa.acc2 = acc2;
                                    aa.acc3 = acc3;
                                    aa.acc4 = acc4;
                                    aa.repeats = C / 8;
                                    aa.tail = C % 8;
                                    aa.wei_stride = sizeof(uint16_t);
                                    aa.wei_blk_stride = aa.wei_stride * 8;
                                    aa.wei = m_wei_packed.data() + base0;
                                    aa.wei2 = m_wei_packed.data() + base1;
                                    aa.wei3 = m_wei_packed.data() + base2;
                                    aa.wei4 = m_wei_packed.data() + base3;
                                    (*m_ip_kernel)(&aa);
                                };
                                const size_t b0 = (((oc0 * KD + kz) * KH + ky) * KW + kx) * m_padded_C;
                                const size_t b1 = has_oc1 ? (((oc1 * KD + kz) * KH + ky) * KW + kx) * m_padded_C : 0;
                                if (has_oc2) {
                                    const size_t b2 = (((oc2 * KD + kz) * KH + ky) * KW + kx) * m_padded_C;
                                    const size_t b3 = has_oc3 ? (((oc3 * KD + kz) * KH + ky) * KW + kx) * m_padded_C : 0;
                                    run_quad2(&acc0, has_oc1 ? &acc1 : nullptr,
                                              &acc2, has_oc3 ? &acc3 : nullptr,
                                              b0, b1, b2, b3);
                                } else {
                                    run_pair2(&acc0, has_oc1 ? &acc1 : nullptr, b0, b1);
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

void JitConv3DExecutor::run_naive_fp16_fallback(const MemoryArgs& memory) {
    // Pure C++ fallback: widen f16->f32, accumulate in f32, store back as f16
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

    const auto* src_p = reinterpret_cast<const uint16_t*>(src->getData());
    const auto* wei_p = reinterpret_cast<const uint16_t*>(wei->getData());
    auto* dst_p = reinterpret_cast<uint16_t*>(dst->getData());

    const size_t src_c_stride = ID * IH * IW;  // elements between channels
    const size_t wei_c_stride = KD * KH * KW;  // elements between weight channels

    auto index_src = [&](size_t n, size_t c, size_t z, size_t y, size_t x) {
        return (((n * C + c) * ID + z) * IH + y) * IW + x;
    };
    auto index_dst = [&](size_t n, size_t c, size_t z, size_t y, size_t x) {
        return (((n * OC + c) * OD + z) * OH + y) * OW + x;
    };
    auto index_wei = [&](size_t oc, size_t c, size_t kz, size_t ky, size_t kx) {
        return ((((oc)*C + c) * KD + kz) * KH + ky) * KW + kx;
    };

    ov::parallel_for2d(N, OD, [&](size_t n, size_t od) {
        const ptrdiff_t iz0 = static_cast<ptrdiff_t>(od) * static_cast<ptrdiff_t>(SD) - PD0;
        for (size_t oh = 0; oh < OH; ++oh) {
            const ptrdiff_t iy0 = static_cast<ptrdiff_t>(oh) * static_cast<ptrdiff_t>(SH) - PH0;
            for (size_t ow = 0; ow < OW; ++ow) {
                const ptrdiff_t ix0 = static_cast<ptrdiff_t>(ow) * static_cast<ptrdiff_t>(SW) - PW0;
                for (size_t oc0 = 0; oc0 < OC; oc0 += 2) {
                    const bool has_oc1 = (oc0 + 1) < OC;
#if defined(__aarch64__)
                    float16x8_t vacc0 = vdupq_n_f16(static_cast<float16_t>(0.0f));
                    float16x8_t vacc1 = vdupq_n_f16(static_cast<float16_t>(0.0f));
#else
                    float acc0_scalar = 0.0f, acc1_scalar = 0.0f;
#endif
                    for (size_t kz = 0; kz < KD; ++kz) {
                        const ptrdiff_t iz = iz0 + static_cast<ptrdiff_t>(kz);
                        if (iz < 0 || iz >= static_cast<ptrdiff_t>(ID)) continue;
                        for (size_t ky = 0; ky < KH; ++ky) {
                            const ptrdiff_t iy = iy0 + static_cast<ptrdiff_t>(ky);
                            if (iy < 0 || iy >= static_cast<ptrdiff_t>(IH)) continue;
                            // Compute kx loop bounds respecting borders
                            for (size_t kx = 0; kx < KW; ++kx) {
                                const ptrdiff_t ix = ix0 + static_cast<ptrdiff_t>(kx);
                                if (ix < 0 || ix >= static_cast<ptrdiff_t>(IW)) continue;
                                const size_t s_base = index_src(n, 0, static_cast<size_t>(iz), static_cast<size_t>(iy), static_cast<size_t>(ix));
                                const size_t w0_base = index_wei(oc0, 0, kz, ky, kx);
                                const size_t w1_base = has_oc1 ? index_wei(oc0 + 1, 0, kz, ky, kx) : 0;
                                // channel loop in blocks of 8
                                size_t c = 0;
                                for (; c + 8 <= C; c += 8) {
#if defined(__aarch64__)
                                    // gather with temporary arrays then load
                                    float16_t src_lanes[8];
                                    float16_t w0_lanes[8];
                                    for (int i = 0; i < 8; ++i) {
                                        src_lanes[i] = *reinterpret_cast<const float16_t*>(&src_p[s_base + (c + i) * src_c_stride]);
                                        w0_lanes[i]  = *reinterpret_cast<const float16_t*>(&wei_p[w0_base + (c + i) * wei_c_stride]);
                                    }
                                    float16x8_t vsrc = vld1q_f16(src_lanes);
                                    float16x8_t vw0  = vld1q_f16(w0_lanes);
                                    vacc0 = vaddq_f16(vacc0, vmulq_f16(vsrc, vw0));
                                    if (has_oc1) {
                                        float16_t w1_lanes[8];
                                        for (int i = 0; i < 8; ++i) {
                                            w1_lanes[i] = *reinterpret_cast<const float16_t*>(&wei_p[w1_base + (c + i) * wei_c_stride]);
                                        }
                                        float16x8_t vw1 = vld1q_f16(w1_lanes);
                                        vacc1 = vaddq_f16(vacc1, vmulq_f16(vsrc, vw1));
                                    }
#else
                                    for (size_t i = 0; i < 8; ++i) {
                                        const float s = static_cast<float>(ov::float16::from_bits(src_p[s_base + (c + i) * src_c_stride]));
                                        const float w0 = static_cast<float>(ov::float16::from_bits(wei_p[w0_base + (c + i) * wei_c_stride]));
                                        acc0_scalar += s * w0;
                                        if (has_oc1) {
                                            const float w1 = static_cast<float>(ov::float16::from_bits(wei_p[w1_base + (c + i) * wei_c_stride]));
                                            acc1_scalar += s * w1;
                                        }
                                    }
#endif
                                }
                                // tail
                                if (c < C) {
#if defined(__aarch64__)
                                    // build tail vectors with zero fill
                                    float16_t src_lanes_t[8] = {0};
                                    float16_t w0_lanes_t[8] = {0};
                                    float16_t w1_lanes_t[8] = {0};
                                    int lane = 0;
                                    for (; c < C && lane < 8; ++c, ++lane) {
                                        src_lanes_t[lane] = *reinterpret_cast<const float16_t*>(&src_p[s_base + c * src_c_stride]);
                                        w0_lanes_t[lane]  = *reinterpret_cast<const float16_t*>(&wei_p[w0_base + c * wei_c_stride]);
                                        if (has_oc1) w1_lanes_t[lane] = *reinterpret_cast<const float16_t*>(&wei_p[w1_base + c * wei_c_stride]);
                                    }
                                    float16x8_t vsrc = vld1q_f16(src_lanes_t);
                                    float16x8_t vw0  = vld1q_f16(w0_lanes_t);
                                    vacc0 = vaddq_f16(vacc0, vmulq_f16(vsrc, vw0));
                                    if (has_oc1) {
                                        float16x8_t vw1 = vld1q_f16(w1_lanes_t);
                                        vacc1 = vaddq_f16(vacc1, vmulq_f16(vsrc, vw1));
                                    }
#else
                                    for (; c < C; ++c) {
                                        const float s = static_cast<float>(ov::float16::from_bits(src_p[s_base + c * src_c_stride]));
                                        const float w0 = static_cast<float>(ov::float16::from_bits(wei_p[w0_base + c * wei_c_stride]));
                                        acc0_scalar += s * w0;
                                        if (has_oc1) {
                                            const float w1 = static_cast<float>(ov::float16::from_bits(wei_p[w1_base + c * wei_c_stride]));
                                            acc1_scalar += s * w1;
                                        }
                                    }
#endif
                                }
                            }
                        }
                    }
                    // horizontal reduce and store
#if defined(__aarch64__)
                    // Convert half accumulators to float and sum lanes
                    float32x4_t lo0 = vcvt_f32_f16(vget_low_f16(vacc0));
                    float32x4_t hi0 = vcvt_f32_f16(vget_high_f16(vacc0));
                    float32x4_t sum0 = vaddq_f32(lo0, hi0);
                    float32x2_t sum0_2 = vadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                    float acc0 = vget_lane_f32(vpadd_f32(sum0_2, sum0_2), 0);
                    uint16_t out0 = ov::float16(acc0).to_bits();
                    dst_p[index_dst(n, oc0, od, oh, ow)] = out0;
                    if (has_oc1) {
                        float32x4_t lo1 = vcvt_f32_f16(vget_low_f16(vacc1));
                        float32x4_t hi1 = vcvt_f32_f16(vget_high_f16(vacc1));
                        float32x4_t sum1 = vaddq_f32(lo1, hi1);
                        float32x2_t sum1_2 = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                        float acc1 = vget_lane_f32(vpadd_f32(sum1_2, sum1_2), 0);
                        dst_p[index_dst(n, oc0 + 1, od, oh, ow)] = ov::float16(acc1).to_bits();
                    }
#else
                    dst_p[index_dst(n, oc0, od, oh, ow)] = ov::float16(acc0_scalar).to_bits();
                    if (has_oc1) dst_p[index_dst(n, oc0 + 1, od, oh, ow)] = ov::float16(acc1_scalar).to_bits();
#endif
                }
            }
        }
    });
}

void JitConv3DExecutor::execute(const MemoryArgs& memory) {
    if (m_is_fp32) {
        run_naive_fp32(memory);
        return;
    }
    // For FP16: prefer JIT kernel when NEON FP16 is available; otherwise use portable fallback
    if (ov::with_cpu_neon_fp16()) {
        run_naive_fp16(memory);
    } else {
        run_naive_fp16_fallback(memory);
    }
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
                                        // single quad-call (oc0..oc3 as available)
                                        const size_t repeats = C / 4;
                                        const size_t tail_c = C % 4;
                                        jit_conv3d_f32_call_args a{};
                                        a.src = src_p + s_base;
                                        a.src_stride = src_c_stride_elems * sizeof(float);
                                        a.src_blk_stride = a.src_stride * 4;
                                        a.acc = &acc0;
                                        a.acc2 = has_oc1 ? &acc1 : nullptr;
                                        a.acc3 = has_oc2 ? &acc2 : nullptr;
                                        a.acc4 = has_oc3 ? &acc3 : nullptr;
                                        a.repeats = repeats;
                                        a.tail = tail_c;
                                        a.kw_cnt = kw_count;
                                        a.src_dx = sizeof(float);
                                        const size_t base0 = (((oc0 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) * KW + static_cast<size_t>(kx_lo)) * m_padded_C_f32;
                                        a.wei = m_wei_packed_f32.data() + base0;
                                        if (has_oc1) {
                                            const size_t base1 = (((oc1 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) * KW + static_cast<size_t>(kx_lo)) * m_padded_C_f32;
                                            a.wei2 = m_wei_packed_f32.data() + base1;
                                        }
                                        if (has_oc2) {
                                            const size_t base2 = (((oc2 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) * KW + static_cast<size_t>(kx_lo)) * m_padded_C_f32;
                                            a.wei3 = m_wei_packed_f32.data() + base2;
                                        }
                                        if (has_oc3) {
                                            const size_t base3 = (((oc3 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) * KW + static_cast<size_t>(kx_lo)) * m_padded_C_f32;
                                            a.wei4 = m_wei_packed_f32.data() + base3;
                                        }
                                        a.wei_stride = sizeof(float);
                                        a.wei_blk_stride = a.wei_stride * 4;
                                        a.wei_dx = m_padded_C_f32 * sizeof(float);
                                        (*m_ip_kernel_f32)(&a);
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
                                        const size_t w2 = has_oc2 ? index_wei(oc2, 0, static_cast<size_t>(kz), static_cast<size_t>(ky), static_cast<size_t>(kx_lo)) : 0;
                                        const size_t w3 = has_oc3 ? index_wei(oc3, 0, static_cast<size_t>(kz), static_cast<size_t>(ky), static_cast<size_t>(kx_lo)) : 0;
                                        jit_conv3d_f32_call_args a{};
                                        a.src = src_p + s_base;
                                        a.src_stride = src_c_stride_elems * sizeof(float);
                                        a.src_blk_stride = a.src_stride * 4;
                                        a.acc = &acc0;
                                        a.acc2 = has_oc1 ? &acc1 : nullptr;
                                        a.acc3 = has_oc2 ? &acc2 : nullptr;
                                        a.acc4 = has_oc3 ? &acc3 : nullptr;
                                        a.repeats = C / 4;
                                        a.tail = C % 4;
                                        a.kw_cnt = kw_count;
                                        a.src_dx = sizeof(float);
                                        a.wei = wei_p + w0;
                                        if (has_oc1) a.wei2 = wei_p + w1;
                                        if (has_oc2) a.wei3 = wei_p + w2;
                                        if (has_oc3) a.wei4 = wei_p + w3;
                                        a.wei_stride = wei_c_stride_elems * sizeof(float);
                                        a.wei_blk_stride = a.wei_stride * 4;
                                        a.wei_dx = sizeof(float);
                                        (*m_ip_kernel_f32)(&a);
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
                                    const size_t base0 = (((oc0 * KD + kz) * KH + ky) * KW + kx) * m_padded_C_f32;
                                    const size_t base1 = has_oc1 ? (((oc1 * KD + kz) * KH + ky) * KW + kx) * m_padded_C_f32 : 0;
                                    const size_t base2 = has_oc2 ? (((oc2 * KD + kz) * KH + ky) * KW + kx) * m_padded_C_f32 : 0;
                                    const size_t base3 = has_oc3 ? (((oc3 * KD + kz) * KH + ky) * KW + kx) * m_padded_C_f32 : 0;
                                    // single quad-call (kw_cnt==1 in generic path)
                                    jit_conv3d_f32_call_args a{};
                                    a.src = src_p + s_base;
                                    a.src_stride = src_c_stride_elems * sizeof(float);
                                    a.src_blk_stride = a.src_stride * 4;
                                    a.acc = &acc0;
                                    a.acc2 = has_oc1 ? &acc1 : nullptr;
                                    a.acc3 = has_oc2 ? &acc2 : nullptr;
                                    a.acc4 = has_oc3 ? &acc3 : nullptr;
                                    a.repeats = C / 4;
                                    a.tail = C % 4;
                                    a.kw_cnt = 1;
                                    a.src_dx = 0;
                                    a.wei = m_wei_packed_f32.data() + base0;
                                    if (has_oc1) a.wei2 = m_wei_packed_f32.data() + base1;
                                    if (has_oc2) a.wei3 = m_wei_packed_f32.data() + base2;
                                    if (has_oc3) a.wei4 = m_wei_packed_f32.data() + base3;
                                    a.wei_stride = sizeof(float);
                                    a.wei_blk_stride = a.wei_stride * 4;
                                    a.wei_dx = 0;
                                    (*m_ip_kernel_f32)(&a);
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
