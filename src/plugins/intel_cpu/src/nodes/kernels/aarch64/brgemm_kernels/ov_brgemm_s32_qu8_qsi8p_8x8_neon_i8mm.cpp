// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/aarch64/brgemm_kernels/int8_brgemm_kernels.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_gen.h>

#include "utils/cpu_utils.hpp"

namespace ov::intel_cpu::aarch64 {

using namespace Xbyak_aarch64;
using namespace dnnl::impl::cpu::aarch64;

jit_int8_brgemm_kernel_8x8_usmmla_packed::jit_int8_brgemm_kernel_8x8_usmmla_packed()
    : jit_generator() {}

void jit_int8_brgemm_kernel_8x8_usmmla_packed::create_ker() {
    jit_generator::create_kernel();
    ker_ = ov::intel_cpu::jit_kernel_cast<ker_t>(jit_ker());
}

void jit_int8_brgemm_kernel_8x8_usmmla_packed::generate() {
    preamble();

    const XReg reg_srcs = abi_param1;
    const XReg reg_wei = abi_param2;
    const XReg reg_dst = abi_param3;
    const XReg reg_k = abi_param4;
    const XReg reg_bias = abi_param5;
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
    const XReg reg_c4 = x22;
    const XReg reg_c5 = x23;
    const XReg reg_c6 = x24;
    const XReg reg_c7 = x25;

    const VReg v_a01 = VReg(0);
    const VReg v_a23 = VReg(1);
    const VReg v_a45 = VReg(2);
    const VReg v_a67 = VReg(3);
    const VReg v_w01 = VReg(6);
    const VReg v_w23 = VReg(7);
    const VReg v_w45 = VReg(8);
    const VReg v_w67 = VReg(9);
    const VReg v_row0_0 = VReg(4);
    const VReg v_row0_1 = VReg(5);
    const VReg v_row1_0 = VReg(10);
    const VReg v_row1_1 = VReg(11);
    const VReg v_tmp0 = VReg(12);
    const VReg v_tmp1 = VReg(13);

    const VReg v_acc00 = VReg(16);
    const VReg v_acc01 = VReg(17);
    const VReg v_acc02 = VReg(18);
    const VReg v_acc03 = VReg(19);
    const VReg v_acc10 = VReg(20);
    const VReg v_acc11 = VReg(21);
    const VReg v_acc12 = VReg(22);
    const VReg v_acc13 = VReg(23);
    const VReg v_acc20 = VReg(24);
    const VReg v_acc21 = VReg(25);
    const VReg v_acc22 = VReg(26);
    const VReg v_acc23 = VReg(27);
    const VReg v_acc30 = VReg(28);
    const VReg v_acc31 = VReg(29);
    const VReg v_acc32 = VReg(30);
    const VReg v_acc33 = VReg(31);

    eor(v_acc00.b, v_acc00.b, v_acc00.b);
    eor(v_acc01.b, v_acc01.b, v_acc01.b);
    eor(v_acc02.b, v_acc02.b, v_acc02.b);
    eor(v_acc03.b, v_acc03.b, v_acc03.b);
    eor(v_acc10.b, v_acc10.b, v_acc10.b);
    eor(v_acc11.b, v_acc11.b, v_acc11.b);
    eor(v_acc12.b, v_acc12.b, v_acc12.b);
    eor(v_acc13.b, v_acc13.b, v_acc13.b);
    eor(v_acc20.b, v_acc20.b, v_acc20.b);
    eor(v_acc21.b, v_acc21.b, v_acc21.b);
    eor(v_acc22.b, v_acc22.b, v_acc22.b);
    eor(v_acc23.b, v_acc23.b, v_acc23.b);
    eor(v_acc30.b, v_acc30.b, v_acc30.b);
    eor(v_acc31.b, v_acc31.b, v_acc31.b);
    eor(v_acc32.b, v_acc32.b, v_acc32.b);
    eor(v_acc33.b, v_acc33.b, v_acc33.b);

    ldr(reg_src0, ptr(reg_srcs));
    ldr(reg_src1, ptr(reg_srcs, 8));
    ldr(reg_src2, ptr(reg_srcs, 16));
    ldr(reg_src3, ptr(reg_srcs, 24));

    mov(reg_c0, reg_dst);
    add(reg_c1, reg_c0, reg_ldc);
    add(reg_c2, reg_c1, reg_ldc);
    add(reg_c3, reg_c2, reg_ldc);
    add(reg_c4, reg_c3, reg_ldc);
    add(reg_c5, reg_c4, reg_ldc);
    add(reg_c6, reg_c5, reg_ldc);
    add(reg_c7, reg_c6, reg_ldc);

    auto emit_usmmla = [&](const VReg4S& vd, const VReg16B& vn, const VReg16B& vm) {
        const uint32_t base = 0x4e80ac00;
        const uint32_t code = base | static_cast<uint32_t>(vd.getIdx()) |
                              (static_cast<uint32_t>(vn.getIdx()) << 5) |
                              (static_cast<uint32_t>(vm.getIdx()) << 16);
        dd(code);
    };

    auto load_a = [&]() {
        ldr(QReg(v_a01.getIdx()), ptr(reg_src0));
        ldr(QReg(v_a23.getIdx()), ptr(reg_src1));
        ldr(QReg(v_a45.getIdx()), ptr(reg_src2));
        ldr(QReg(v_a67.getIdx()), ptr(reg_src3));
    };

    auto advance_src = [&]() {
        add(reg_src0, reg_src0, 16);
        add(reg_src1, reg_src1, 16);
        add(reg_src2, reg_src2, 16);
        add(reg_src3, reg_src3, 16);
    };

    auto prefetch_next = [&]() {
        prfm(PLDL1KEEP, ptr(reg_wei, 256));
        prfm(PLDL1KEEP, ptr(reg_src0, 128));
        prfm(PLDL1KEEP, ptr(reg_src1, 128));
        prfm(PLDL1KEEP, ptr(reg_src2, 128));
        prfm(PLDL1KEEP, ptr(reg_src3, 128));
    };

    auto compute8 = [&]() {
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
        emit_usmmla(v_acc20.s, v_a45.b, v_w01.b);
        emit_usmmla(v_acc21.s, v_a45.b, v_w23.b);
        emit_usmmla(v_acc22.s, v_a45.b, v_w45.b);
        emit_usmmla(v_acc23.s, v_a45.b, v_w67.b);
        emit_usmmla(v_acc30.s, v_a67.b, v_w01.b);
        emit_usmmla(v_acc31.s, v_a67.b, v_w23.b);
        emit_usmmla(v_acc32.s, v_a67.b, v_w45.b);
        emit_usmmla(v_acc33.s, v_a67.b, v_w67.b);

        advance_src();
        add(reg_wei, reg_wei, 64);
    };

    Label loop32;
    Label loop16;
    Label loop8;
    Label done;

    cbz(reg_k, done);

    cmp(reg_k, 32);
    b(LT, loop16);

    L(loop32);
    prefetch_next();
    compute8();
    compute8();
    compute8();
    compute8();
    subs(reg_k, reg_k, 32);
    cmp(reg_k, 32);
    b(GE, loop32);

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
    Label store_no_accum_bias;
    Label store_accum;
    Label store_accum_bias;
    Label store_done;

    auto store_pair_acc = [&](const XReg& reg_lo,
                              const XReg& reg_hi,
                              const VReg& acc0,
                              const VReg& acc1,
                              const VReg& acc2,
                              const VReg& acc3,
                              const bool add_bias) {
        zip1(v_row0_0.d, acc0.d, acc1.d);
        zip1(v_row0_1.d, acc2.d, acc3.d);
        zip2(v_row1_0.d, acc0.d, acc1.d);
        zip2(v_row1_1.d, acc2.d, acc3.d);

        if (add_bias) {
            ldp(QReg(v_tmp0.getIdx()), QReg(v_tmp1.getIdx()), ptr(reg_bias));
            add(v_row0_0.s, v_row0_0.s, v_tmp0.s);
            add(v_row0_1.s, v_row0_1.s, v_tmp1.s);
            add(v_row1_0.s, v_row1_0.s, v_tmp0.s);
            add(v_row1_1.s, v_row1_1.s, v_tmp1.s);
        }

        ldp(QReg(v_tmp0.getIdx()), QReg(v_tmp1.getIdx()), ptr(reg_lo));
        add(v_tmp0.s, v_tmp0.s, v_row0_0.s);
        add(v_tmp1.s, v_tmp1.s, v_row0_1.s);
        stp(QReg(v_tmp0.getIdx()), QReg(v_tmp1.getIdx()), ptr(reg_lo));

        ldp(QReg(v_tmp0.getIdx()), QReg(v_tmp1.getIdx()), ptr(reg_hi));
        add(v_tmp0.s, v_tmp0.s, v_row1_0.s);
        add(v_tmp1.s, v_tmp1.s, v_row1_1.s);
        stp(QReg(v_tmp0.getIdx()), QReg(v_tmp1.getIdx()), ptr(reg_hi));
    };

    auto store_pair_noacc = [&](const XReg& reg_lo,
                                const XReg& reg_hi,
                                const VReg& acc0,
                                const VReg& acc1,
                                const VReg& acc2,
                                const VReg& acc3,
                                const bool add_bias) {
        zip1(v_row0_0.d, acc0.d, acc1.d);
        zip1(v_row0_1.d, acc2.d, acc3.d);
        zip2(v_row1_0.d, acc0.d, acc1.d);
        zip2(v_row1_1.d, acc2.d, acc3.d);

        if (add_bias) {
            ldp(QReg(v_tmp0.getIdx()), QReg(v_tmp1.getIdx()), ptr(reg_bias));
            add(v_row0_0.s, v_row0_0.s, v_tmp0.s);
            add(v_row0_1.s, v_row0_1.s, v_tmp1.s);
            add(v_row1_0.s, v_row1_0.s, v_tmp0.s);
            add(v_row1_1.s, v_row1_1.s, v_tmp1.s);
        }

        stp(QReg(v_row0_0.getIdx()), QReg(v_row0_1.getIdx()), ptr(reg_lo));
        stp(QReg(v_row1_0.getIdx()), QReg(v_row1_1.getIdx()), ptr(reg_hi));
    };

    cbz(reg_accum, store_no_accum);
    cbz(reg_bias, store_accum);
    b(store_accum_bias);

    L(store_accum);
    store_pair_acc(reg_c0, reg_c1, v_acc00, v_acc01, v_acc02, v_acc03, false);
    store_pair_acc(reg_c2, reg_c3, v_acc10, v_acc11, v_acc12, v_acc13, false);
    store_pair_acc(reg_c4, reg_c5, v_acc20, v_acc21, v_acc22, v_acc23, false);
    store_pair_acc(reg_c6, reg_c7, v_acc30, v_acc31, v_acc32, v_acc33, false);
    b(store_done);

    L(store_accum_bias);
    store_pair_acc(reg_c0, reg_c1, v_acc00, v_acc01, v_acc02, v_acc03, true);
    store_pair_acc(reg_c2, reg_c3, v_acc10, v_acc11, v_acc12, v_acc13, true);
    store_pair_acc(reg_c4, reg_c5, v_acc20, v_acc21, v_acc22, v_acc23, true);
    store_pair_acc(reg_c6, reg_c7, v_acc30, v_acc31, v_acc32, v_acc33, true);
    b(store_done);

    L(store_no_accum);
    cbnz(reg_bias, store_no_accum_bias);
    store_pair_noacc(reg_c0, reg_c1, v_acc00, v_acc01, v_acc02, v_acc03, false);
    store_pair_noacc(reg_c2, reg_c3, v_acc10, v_acc11, v_acc12, v_acc13, false);
    store_pair_noacc(reg_c4, reg_c5, v_acc20, v_acc21, v_acc22, v_acc23, false);
    store_pair_noacc(reg_c6, reg_c7, v_acc30, v_acc31, v_acc32, v_acc33, false);
    b(store_done);

    L(store_no_accum_bias);
    store_pair_noacc(reg_c0, reg_c1, v_acc00, v_acc01, v_acc02, v_acc03, true);
    store_pair_noacc(reg_c2, reg_c3, v_acc10, v_acc11, v_acc12, v_acc13, true);
    store_pair_noacc(reg_c4, reg_c5, v_acc20, v_acc21, v_acc22, v_acc23, true);
    store_pair_noacc(reg_c6, reg_c7, v_acc30, v_acc31, v_acc32, v_acc33, true);

    L(store_done);

    L(done);
    postamble();
}

}  // namespace ov::intel_cpu::aarch64
