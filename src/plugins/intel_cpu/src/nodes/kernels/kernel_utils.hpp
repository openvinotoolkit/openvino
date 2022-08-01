// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/jit_generator.hpp"
#include <dnnl_types.h>


namespace ov {
namespace intel_cpu {

class jitKernelBase: public dnnl::impl::cpu::x64::jit_generator {
protected:

inline bool isValidIsa(dnnl::impl::cpu::x64::cpu_isa_t isa) {
    return is_subset(isa, dnnl::impl::cpu::x64::isa_all) && dnnl::impl::cpu::x64::mayiuse(isa);
}

void uni_vfmsub132ps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
                     const Xbyak::Operand &op) {
    // Note: x1 gets overriden by x1*op
    // This is incorrect if x1 == x2
    if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
        vfmsub132ps(x1, x2, op);
    } else if (isValidIsa(dnnl::impl::cpu::x64::avx)) {
        assert(x1.getIdx() != x2.getIdx());
        vmulps(x1, x1, op);
        vsubps(x1, x1, x2);
    } else {
        assert(x1.getIdx() != x2.getIdx());
        mulps(x1, op);
        subps(x1, x2);
    }
}

void uni_vfmsub231ps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2, const Xbyak::Operand &op) {
    // Note: x1 gets overriden by x1*x2
    // This is incorrect if x1 == op
    if (isValidIsa(dnnl::impl::cpu::x64::avx2))
        vfmsub231ps(x1, x2, op);
    else if (isValidIsa(dnnl::impl::cpu::x64::avx)) {
        assert(!x1.isEqualIfNotInherited(op));
        vmulps(x1, x1, x2);
        vsubps(x1, x1, op);
    } else {
        assert(!x1.isEqualIfNotInherited(op));
        mulps(x1, x2);
        subps(x1, op);
    }
}

void uni_kmovd(const Xbyak::Opmask& kDst, const Xbyak::Opmask& kSrc) {
    kmovq(kDst, kSrc);
}

void uni_kmovd(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrc) {
    uni_vmovups(vDst, vSrc);
}

void uni_kxnorw(const Xbyak::Opmask& kDst, const Xbyak::Opmask& kSrc1, const Xbyak::Opmask& kSrc2) {
    kxnorw(kDst, kSrc1, kSrc2);
}

void uni_kxnorw(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrc1, const Xbyak::Xmm& vSrc2) {
    uni_vpxor(vDst, vSrc1, vSrc2);
    if (dnnl::impl::cpu::x64::is_subset(dnnl::impl::cpu::x64::avx, dnnl::impl::cpu::x64::isa_all) &&
              dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx)) {
        vandnps(vDst, vSrc1, vSrc2);
    } else {
        andnps(vDst, vSrc1);
    }
}

void uni_vpgatherdd(const Xbyak::Xmm& vDst, const Xbyak::Address& srcAddr, const Xbyak::Xmm& vMask) {
    vpgatherdd(vDst, srcAddr, vMask);
}

void uni_vpgatherdd(const Xbyak::Zmm& vDst, const Xbyak::Address& srcAddr, const Xbyak::Opmask& kMask) {
    vpgatherdd(vDst | kMask, srcAddr);
}

void uni_vpermd(const Xbyak::Zmm& vDst, const Xbyak::Zmm& vMask, const Xbyak::Operand& src) {
    vpermd(vDst, vMask, src);
}

void uni_vpinsrd(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrc1, const Xbyak::Operand& vSrc2, const int imm) {
    if (isValidIsa(dnnl::impl::cpu::x64::avx)) {
        vpinsrd(vDst, vSrc1, vSrc2, imm);
    } else {
        if (vSrc1.getIdx() != vSrc2.getIdx()) movdqa(vSrc1, vSrc2);
        pinsrd(vDst, vSrc2, imm);
    }
}

void fillRestWorkMask(const Xbyak::Opmask& kDstMask,
                      const Xbyak::Zmm& zAux,
                      const Xbyak::Reg64& rWorkRest,
                      const Xbyak::Reg64& rAux0,
                      const Xbyak::Reg64& rAux1) {
    Xbyak::Label lKmov;
    Xbyak::Reg32 rOnes(rAux1.getIdx());
    const uint32_t typeSize = 4;
    const uint64_t elPerVec = dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::avx512_core>::vlen / typeSize;

    mov(rOnes, 0x0000FFFF);
    cmp(rWorkRest, elPerVec);
    jge(lKmov);
    {
        Xbyak::Reg32 rShift(rAux0.getIdx());
        mov(rShift, elPerVec);
        sub(rShift, rWorkRest);
        shrx(rOnes, rOnes, rShift);
    }
    L(lKmov);
    kmovw(kDstMask, rOnes);
}

void partialLoad32(const Xbyak::Ymm& vDst,
                   const Xbyak::Reg64& rSrc,
                   const Xbyak::Ymm& vAux,
                   const Xbyak::Reg64& rLoadNum,
                   const Xbyak::Reg64& rAux) {
    const uint8_t typeSize = 4;
    const uint8_t elPerVec = dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::avx>::vlen / typeSize;
    Xbyak::Label lLoopEnd0, lLoopEnd1;
    mov(rAux, rLoadNum);
    Xbyak::Xmm xmmAux(vDst.getIdx());
    uni_vpxor(vDst, vDst, vDst);
    for (uint8_t i = 0; i < elPerVec / 2; i++) {
        cmp(rAux, 0);
        je(lLoopEnd0, T_NEAR);

        uni_vpinsrd(xmmAux, xmmAux, ptr[rSrc + i * typeSize], i);

        dec(rAux);
    }
    // vperm2f128(01);
    xmmAux = Xbyak::Xmm(vAux.getIdx());
    uni_vpxor(xmmAux, xmmAux, xmmAux);
    for (uint8_t i = 0; i < elPerVec / 2; i++) {
        cmp(rAux, 0);
        je(lLoopEnd1, T_NEAR);

        uni_vpinsrd(xmmAux, xmmAux, ptr[rSrc + i * typeSize], i);

        dec(rAux);
    }
    L(lLoopEnd1);
    vinsertf128(vDst, vDst, xmmAux, 1);
    L(lLoopEnd0);
    // vperm2f128(10);
}

void partialLoad32(const Xbyak::Xmm& vDst,
                   const Xbyak::Reg64& rSrc,
                   const Xbyak::Xmm& vAux,
                   const Xbyak::Reg64& rLoadNum,
                   const Xbyak::Reg64& rAux) {
    const uint8_t typeSize = 4;
    const uint8_t elPerVec = dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::sse41>::vlen / typeSize;
    Xbyak::Label lLoopEnd0;
    mov(rAux, rLoadNum);
    Xbyak::Xmm xmmAux(vDst.getIdx());
    uni_vpxor(vDst, vDst, vDst);
    for (uint8_t i = 0; i < elPerVec; i++) {
        cmp(rAux, 0);
        je(lLoopEnd0, T_NEAR);

        uni_vpinsrd(xmmAux, xmmAux, ptr[rSrc + i * typeSize], i);

        dec(rAux);
    }
    L(lLoopEnd0);
}

// Makes gather from memory under the vReadMask and writes to the XMM/m128 under the vWriteMask
// It can fill in values not read from the source with zero.
void maskMov32(const Xbyak::Operand& opDst,
               const Xbyak::Operand& opSrc,
               const Xbyak::Xmm&     vReadMask,
               const Xbyak::Xmm&     vWriteMask,
               const Xbyak::Xmm&     vSrcShift,
               const Xbyak::Xmm&     vAux,
               const Xbyak::Reg64&   rAux,
               const bool useMask  = false,
               const bool zeroMask = false) {
    Xbyak::Reg32 r32Aux = Xbyak::Reg32(rAux.getIdx());
    const uint8_t typeSize = 4;

    for (uint8_t i = 0; i < 4; i++) {
        Xbyak::Label lLoopNext, lZeroMask;
        if (useMask) {
            uni_vpextrd(r32Aux, vReadMask, i);
            cmp(r32Aux, 0);
            je(lZeroMask, T_NEAR);
        }
        uni_vpextrd(r32Aux, vSrcShift, i);
        if (opDst.isXMM()) {
            Xbyak::Xmm vDst = Xbyak::Xmm(opDst.getIdx());
            uni_vpinsrd(vDst, vDst, ptr[opSrc.getReg() + rAux], i << 4);
        } else if (opDst.isREG()) {
            mov(rAux, ptr[opSrc.getReg() + rAux]);
            mov(ptr[opDst.getReg() + i * typeSize], rAux);
        }
        jmp(lLoopNext, T_NEAR);
        L(lZeroMask);
        if (zeroMask) {
            if (opDst.isXMM()) {
                Xbyak::Xmm vDst = Xbyak::Xmm(opDst.getIdx());
                uni_vpinsrd(vDst, vDst, r32Aux, i << 4);
            } else if (opDst.isREG()) {
                mov(ptr[opDst.getReg() + i * typeSize], rAux);
            }
        }
        L(lLoopNext);
    } // use VMASKMOVDQU?
}

// Makes gather from memory under the vReadMask and writes to the YMM/m256 under the vWriteMask
// It can fill in values not read from the source with zero.
void maskMov32(const Xbyak::Operand& opDst,
               const Xbyak::Operand& opSrc,
               const Xbyak::Ymm&     vReadMask,
               const Xbyak::Ymm&     vWriteMask,
               const Xbyak::Ymm&     vSrcShift,
               const Xbyak::Ymm&     vAux,
               const Xbyak::Reg64&   rAux,
               const bool useMask  = false,
               const bool zeroMask = false) {
    if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
        if (opDst.isYMM()) {
            Xbyak::Ymm vDst = Xbyak::Ymm(opDst.getIdx());
            if (zeroMask)
                uni_vpxor(vDst, vDst, vDst);
            uni_vpgatherdd(vDst, ptr[vSrcShift + vSrcShift], vReadMask);
        } else if (opDst.isREG()) {
            if (zeroMask)
                uni_vpxor(vAux, vAux, vAux);
            uni_vpgatherdd(vAux, ptr[vSrcShift + vSrcShift], vReadMask);
            if (zeroMask)
                uni_vmovups(ptr[opDst.getReg()], vAux);
            else
                uni_vmovups_tail(ptr[opDst.getReg()], vWriteMask, vAux);
        }
    } else {
        Xbyak::Xmm xmmReadMask  = Xbyak::Xmm(vReadMask.getIdx()),
                   xmmWriteMask = Xbyak::Xmm(vWriteMask.getIdx()),
                   xmmSrcShft   = Xbyak::Xmm(vSrcShift.getIdx());
        for (uint8_t i = 0; i < 2; i++) {
            Xbyak::Xmm xmmAux = Xbyak::Xmm(vAux.getIdx());
            maskMov32(opDst, opSrc, xmmReadMask, xmmWriteMask, xmmSrcShft, xmmAux, rAux, useMask);
            if (opDst.isYMM()) {
                Xbyak::Ymm vDst = Xbyak::Ymm(opDst.getIdx());
                vperm2f128(vDst, vDst, vDst, 0x1);
            }
            if (useMask)
                vperm2f128(vReadMask, vReadMask, vReadMask, 0x1);
            if (zeroMask)
                vperm2f128(vWriteMask, vWriteMask, vWriteMask, 0x1);
            vperm2f128(vSrcShift, vSrcShift, vSrcShift, 0x1);
        }
    }
}
};

}
}
