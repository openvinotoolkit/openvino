// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/aarch64/brgemm_kernels/int8_brgemm_kernels.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_gen.h>

#include "utils/cpu_utils.hpp"

namespace ov::intel_cpu::aarch64 {

using namespace Xbyak_aarch64;
using namespace dnnl::impl::cpu::aarch64;

jit_int8_brgemm_kernel_4x16_usmmla_packed::jit_int8_brgemm_kernel_4x16_usmmla_packed(bool interleaved)
    : jit_generator(),
      interleaved_(interleaved) {}

void jit_int8_brgemm_kernel_4x16_usmmla_packed::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_4x16_usmmla_packed::generate() {
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

    const VReg v_a01 = VReg(0);
    const VReg v_a23 = VReg(1);
    const VReg v_tmp0 = VReg(2);
    const VReg v_tmp1 = VReg(3);
    const VReg v_w01 = VReg(4);
    const VReg v_w23 = VReg(5);
    const VReg v_w45 = VReg(6);
    const VReg v_w67 = VReg(7);
    const VReg v_w89 = VReg(8);
    const VReg v_wab = VReg(9);
    const VReg v_wcd = VReg(10);
    const VReg v_wef = VReg(11);

    const VReg v_acc00 = VReg(16);
    const VReg v_acc01 = VReg(17);
    const VReg v_acc02 = VReg(18);
    const VReg v_acc03 = VReg(19);
    const VReg v_acc04 = VReg(20);
    const VReg v_acc05 = VReg(21);
    const VReg v_acc06 = VReg(22);
    const VReg v_acc07 = VReg(23);
    const VReg v_acc10 = VReg(24);
    const VReg v_acc11 = VReg(25);
    const VReg v_acc12 = VReg(26);
    const VReg v_acc13 = VReg(27);
    const VReg v_acc14 = VReg(28);
    const VReg v_acc15 = VReg(29);
    const VReg v_acc16 = VReg(30);
    const VReg v_acc17 = VReg(31);

    eor(v_acc00.b, v_acc00.b, v_acc00.b);
    eor(v_acc01.b, v_acc01.b, v_acc01.b);
    eor(v_acc02.b, v_acc02.b, v_acc02.b);
    eor(v_acc03.b, v_acc03.b, v_acc03.b);
    eor(v_acc04.b, v_acc04.b, v_acc04.b);
    eor(v_acc05.b, v_acc05.b, v_acc05.b);
    eor(v_acc06.b, v_acc06.b, v_acc06.b);
    eor(v_acc07.b, v_acc07.b, v_acc07.b);
    eor(v_acc10.b, v_acc10.b, v_acc10.b);
    eor(v_acc11.b, v_acc11.b, v_acc11.b);
    eor(v_acc12.b, v_acc12.b, v_acc12.b);
    eor(v_acc13.b, v_acc13.b, v_acc13.b);
    eor(v_acc14.b, v_acc14.b, v_acc14.b);
    eor(v_acc15.b, v_acc15.b, v_acc15.b);
    eor(v_acc16.b, v_acc16.b, v_acc16.b);
    eor(v_acc17.b, v_acc17.b, v_acc17.b);

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
    };

    auto compute8 = [&]() {
        load_a();

        ldp(QReg(v_w01.getIdx()), QReg(v_w23.getIdx()), ptr(reg_wei));
        ldp(QReg(v_w45.getIdx()), QReg(v_w67.getIdx()), ptr(reg_wei, 32));
        ldp(QReg(v_w89.getIdx()), QReg(v_wab.getIdx()), ptr(reg_wei, 64));
        ldp(QReg(v_wcd.getIdx()), QReg(v_wef.getIdx()), ptr(reg_wei, 96));

        emit_usmmla(v_acc00.s, v_a01.b, v_w01.b);
        emit_usmmla(v_acc01.s, v_a01.b, v_w23.b);
        emit_usmmla(v_acc02.s, v_a01.b, v_w45.b);
        emit_usmmla(v_acc03.s, v_a01.b, v_w67.b);
        emit_usmmla(v_acc04.s, v_a01.b, v_w89.b);
        emit_usmmla(v_acc05.s, v_a01.b, v_wab.b);
        emit_usmmla(v_acc06.s, v_a01.b, v_wcd.b);
        emit_usmmla(v_acc07.s, v_a01.b, v_wef.b);
        emit_usmmla(v_acc10.s, v_a23.b, v_w01.b);
        emit_usmmla(v_acc11.s, v_a23.b, v_w23.b);
        emit_usmmla(v_acc12.s, v_a23.b, v_w45.b);
        emit_usmmla(v_acc13.s, v_a23.b, v_w67.b);
        emit_usmmla(v_acc14.s, v_a23.b, v_w89.b);
        emit_usmmla(v_acc15.s, v_a23.b, v_wab.b);
        emit_usmmla(v_acc16.s, v_a23.b, v_wcd.b);
        emit_usmmla(v_acc17.s, v_a23.b, v_wef.b);

        advance_src();
        add(reg_wei, reg_wei, 128);
    };

    Label loop64;
    Label loop32;
    Label loop32_done;
    Label loop16;
    Label loop8;
    Label done;

    cbz(reg_k, done);

    cmp(reg_k, 64);
    b(LT, loop32);

    L(loop64);
    prefetch_next();
    compute8();
    compute8();
    compute8();
    compute8();
    compute8();
    compute8();
    compute8();
    compute8();
    subs(reg_k, reg_k, 64);
    cmp(reg_k, 64);
    b(GE, loop64);

    cmp(reg_k, 32);
    b(LT, loop32_done);

    L(loop32);
    prefetch_next();
    compute8();
    compute8();
    compute8();
    compute8();
    subs(reg_k, reg_k, 32);
    cmp(reg_k, 32);
    b(GE, loop32);

    L(loop32_done);
    cmp(reg_k, 16);
    b(LT, loop8);

    L(loop16);
    prefetch_next();
    compute8();
    compute8();

    subs(reg_k, reg_k, 16);
    cmp(reg_k, 16);
    b(GE, loop16);

    L(loop8);
    cbz(reg_k, done);
    prefetch_next();
    compute8();

    subs(reg_k, reg_k, 8);
    // reg_k is expected to hit zero here

    Label store_no_accum;
    Label store_accum;
    Label store_done;

    auto store_acc = [&](const XReg& reg_c, int offset, const VReg& v_acc, bool upper) {
        ldr(DReg(v_tmp0.getIdx()), ptr(reg_c, offset));
        if (upper) {
            mov(v_tmp1.d[0], v_acc.d[1]);
            add(v_tmp0.s, v_tmp0.s, v_tmp1.s);
        } else {
            add(v_tmp0.s, v_tmp0.s, v_acc.s);
        }
        str(DReg(v_tmp0.getIdx()), ptr(reg_c, offset));
    };

    auto store_noacc = [&](const XReg& reg_c, int offset, const VReg& v_acc, bool upper) {
        if (upper) {
            mov(v_tmp1.d[0], v_acc.d[1]);
            str(DReg(v_tmp1.getIdx()), ptr(reg_c, offset));
        } else {
            str(DReg(v_acc.getIdx()), ptr(reg_c, offset));
        }
    };

    cbz(reg_accum, store_no_accum);

    L(store_accum);
    store_acc(reg_c0, 0, v_acc00, false);
    store_acc(reg_c0, 8, v_acc01, false);
    store_acc(reg_c0, 16, v_acc02, false);
    store_acc(reg_c0, 24, v_acc03, false);
    store_acc(reg_c0, 32, v_acc04, false);
    store_acc(reg_c0, 40, v_acc05, false);
    store_acc(reg_c0, 48, v_acc06, false);
    store_acc(reg_c0, 56, v_acc07, false);

    store_acc(reg_c1, 0, v_acc00, true);
    store_acc(reg_c1, 8, v_acc01, true);
    store_acc(reg_c1, 16, v_acc02, true);
    store_acc(reg_c1, 24, v_acc03, true);
    store_acc(reg_c1, 32, v_acc04, true);
    store_acc(reg_c1, 40, v_acc05, true);
    store_acc(reg_c1, 48, v_acc06, true);
    store_acc(reg_c1, 56, v_acc07, true);

    store_acc(reg_c2, 0, v_acc10, false);
    store_acc(reg_c2, 8, v_acc11, false);
    store_acc(reg_c2, 16, v_acc12, false);
    store_acc(reg_c2, 24, v_acc13, false);
    store_acc(reg_c2, 32, v_acc14, false);
    store_acc(reg_c2, 40, v_acc15, false);
    store_acc(reg_c2, 48, v_acc16, false);
    store_acc(reg_c2, 56, v_acc17, false);

    store_acc(reg_c3, 0, v_acc10, true);
    store_acc(reg_c3, 8, v_acc11, true);
    store_acc(reg_c3, 16, v_acc12, true);
    store_acc(reg_c3, 24, v_acc13, true);
    store_acc(reg_c3, 32, v_acc14, true);
    store_acc(reg_c3, 40, v_acc15, true);
    store_acc(reg_c3, 48, v_acc16, true);
    store_acc(reg_c3, 56, v_acc17, true);
    b(store_done);

    L(store_no_accum);
    store_noacc(reg_c0, 0, v_acc00, false);
    store_noacc(reg_c0, 8, v_acc01, false);
    store_noacc(reg_c0, 16, v_acc02, false);
    store_noacc(reg_c0, 24, v_acc03, false);
    store_noacc(reg_c0, 32, v_acc04, false);
    store_noacc(reg_c0, 40, v_acc05, false);
    store_noacc(reg_c0, 48, v_acc06, false);
    store_noacc(reg_c0, 56, v_acc07, false);

    store_noacc(reg_c1, 0, v_acc00, true);
    store_noacc(reg_c1, 8, v_acc01, true);
    store_noacc(reg_c1, 16, v_acc02, true);
    store_noacc(reg_c1, 24, v_acc03, true);
    store_noacc(reg_c1, 32, v_acc04, true);
    store_noacc(reg_c1, 40, v_acc05, true);
    store_noacc(reg_c1, 48, v_acc06, true);
    store_noacc(reg_c1, 56, v_acc07, true);

    store_noacc(reg_c2, 0, v_acc10, false);
    store_noacc(reg_c2, 8, v_acc11, false);
    store_noacc(reg_c2, 16, v_acc12, false);
    store_noacc(reg_c2, 24, v_acc13, false);
    store_noacc(reg_c2, 32, v_acc14, false);
    store_noacc(reg_c2, 40, v_acc15, false);
    store_noacc(reg_c2, 48, v_acc16, false);
    store_noacc(reg_c2, 56, v_acc17, false);

    store_noacc(reg_c3, 0, v_acc10, true);
    store_noacc(reg_c3, 8, v_acc11, true);
    store_noacc(reg_c3, 16, v_acc12, true);
    store_noacc(reg_c3, 24, v_acc13, true);
    store_noacc(reg_c3, 32, v_acc14, true);
    store_noacc(reg_c3, 40, v_acc15, true);
    store_noacc(reg_c3, 48, v_acc16, true);
    store_noacc(reg_c3, 56, v_acc17, true);

    L(store_done);

    L(done);
    postamble();
}

}  // namespace ov::intel_cpu::aarch64
