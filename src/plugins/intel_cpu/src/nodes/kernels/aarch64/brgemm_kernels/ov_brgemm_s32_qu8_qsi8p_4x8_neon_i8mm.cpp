// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/aarch64/brgemm_kernels/int8_brgemm_kernels.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_gen.h>

#include "utils/cpu_utils.hpp"

namespace ov::intel_cpu::aarch64 {

using namespace Xbyak_aarch64;
using namespace dnnl::impl::cpu::aarch64;

jit_int8_brgemm_kernel_4x8_usmmla_packed::jit_int8_brgemm_kernel_4x8_usmmla_packed(bool interleaved)
    : jit_generator(),
      interleaved_(interleaved) {}

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
    const WReg reg_tmp = w9;
    const WReg reg_c_acc = w0;

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

    auto load_a = [&]() {
        if (interleaved_) {
            ldr(QReg(v_a01.getIdx()), ptr(reg_src0));
            ldr(QReg(v_a23.getIdx()), ptr(reg_src1));
        } else {
            ldr(DReg(v_a01.getIdx()), ptr(reg_src0));
            ldr(DReg(v_tmp0.getIdx()), ptr(reg_src1));
            ins(v_a01.d[1], v_tmp0.d[0]);
            ldr(DReg(v_a23.getIdx()), ptr(reg_src2));
            ldr(DReg(v_tmp1.getIdx()), ptr(reg_src3));
            ins(v_a23.d[1], v_tmp1.d[0]);
        }
    };

    auto advance_src = [&]() {
        if (interleaved_) {
            add(reg_src0, reg_src0, 16);
            add(reg_src1, reg_src1, 16);
        } else {
            add(reg_src0, reg_src0, 8);
            add(reg_src1, reg_src1, 8);
            add(reg_src2, reg_src2, 8);
            add(reg_src3, reg_src3, 8);
        }
    };

    auto prefetch_next = [&]() {
        prfm(PLDL1KEEP, ptr(reg_wei, 256));
        if (interleaved_) {
            prfm(PLDL1KEEP, ptr(reg_src0, 64));
            prfm(PLDL1KEEP, ptr(reg_src1, 64));
        } else {
            prfm(PLDL1KEEP, ptr(reg_src0, 32));
            prfm(PLDL1KEEP, ptr(reg_src1, 32));
            prfm(PLDL1KEEP, ptr(reg_src2, 32));
            prfm(PLDL1KEEP, ptr(reg_src3, 32));
        }
    };

    Label loop16;
    Label loop8;
    Label done;

    cbz(reg_k, done);

    cmp(reg_k, 16);
    b(LT, loop8);

    L(loop16);
    prefetch_next();
    load_a();

    ldp(QReg(v_w01.getIdx()), QReg(v_w23.getIdx()), ptr(reg_wei));
    ldp(QReg(v_w45.getIdx()), QReg(v_w67.getIdx()), ptr(reg_wei, 32));

    emit_usmmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_usmmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_usmmla(v_acc02.s, v_a01.b, v_w45.b);
    emit_usmmla(v_acc03.s, v_a01.b, v_w67.b);
    emit_usmmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_usmmla(v_acc11.s, v_a23.b, v_w23.b);
    emit_usmmla(v_acc12.s, v_a23.b, v_w45.b);
    emit_usmmla(v_acc13.s, v_a23.b, v_w67.b);

    advance_src();
    add(reg_wei, reg_wei, 64);

    load_a();

    ldp(QReg(v_w01.getIdx()), QReg(v_w23.getIdx()), ptr(reg_wei));
    ldp(QReg(v_w45.getIdx()), QReg(v_w67.getIdx()), ptr(reg_wei, 32));

    emit_usmmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_usmmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_usmmla(v_acc02.s, v_a01.b, v_w45.b);
    emit_usmmla(v_acc03.s, v_a01.b, v_w67.b);
    emit_usmmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_usmmla(v_acc11.s, v_a23.b, v_w23.b);
    emit_usmmla(v_acc12.s, v_a23.b, v_w45.b);
    emit_usmmla(v_acc13.s, v_a23.b, v_w67.b);

    advance_src();
    add(reg_wei, reg_wei, 64);

    subs(reg_k, reg_k, 16);
    cmp(reg_k, 16);
    b(GE, loop16);

    L(loop8);
    cbz(reg_k, done);
    prefetch_next();
    load_a();

    ldp(QReg(v_w01.getIdx()), QReg(v_w23.getIdx()), ptr(reg_wei));
    ldp(QReg(v_w45.getIdx()), QReg(v_w67.getIdx()), ptr(reg_wei, 32));

    emit_usmmla(v_acc00.s, v_a01.b, v_w01.b);
    emit_usmmla(v_acc01.s, v_a01.b, v_w23.b);
    emit_usmmla(v_acc02.s, v_a01.b, v_w45.b);
    emit_usmmla(v_acc03.s, v_a01.b, v_w67.b);
    emit_usmmla(v_acc10.s, v_a23.b, v_w01.b);
    emit_usmmla(v_acc11.s, v_a23.b, v_w23.b);
    emit_usmmla(v_acc12.s, v_a23.b, v_w45.b);
    emit_usmmla(v_acc13.s, v_a23.b, v_w67.b);

    advance_src();
    add(reg_wei, reg_wei, 64);

    subs(reg_k, reg_k, 8);
    // reg_k is expected to hit zero here

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

}  // namespace ov::intel_cpu::aarch64
