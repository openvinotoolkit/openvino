// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel_base.hpp"

using namespace ov;
using namespace intel_cpu;
using namespace dnnl::impl::cpu;


void JitKernelBase::uni_vfmsub132ps(const Xbyak::Xmm& vDst,
                                    const Xbyak::Xmm& vSrc,
                                    const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vfmsub132ps(vDst, vSrc, op);
    } else if (isValidIsa(x64::avx)) {
        assert(vDst.getIdx() != vSrc.getIdx());
        vmulps(vDst, vDst, op);
        vsubps(vDst, vDst, vSrc);
    } else {
        assert(vDst.getIdx() != vSrc.getIdx());
        mulps(vDst, op);
        subps(vDst, vSrc);
    }
}

void JitKernelBase::uni_vfnmadd132ps(const Xbyak::Xmm& vDst,
                                     const Xbyak::Xmm& vSrc,
                                     const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vfnmadd132ps(vDst, vSrc, op);
    } else if (isValidIsa(x64::avx)) {
        assert(vDst.getIdx() != vSrc.getIdx());
        vmulps(vDst, vDst, op);
        vsubps(vDst, vSrc, vDst);
    } else {
        assert(vDst.getIdx() != vSrc.getIdx());
        mulps(vDst, op);
        subps(vSrc, vDst);
        movups(vDst, vSrc);
    }
}

void JitKernelBase::uni_vfmsub231ps(const Xbyak::Xmm& vDst,
                                    const Xbyak::Xmm& vSrc,
                                    const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vfmsub231ps(vDst, vSrc, op);
    } else if (isValidIsa(x64::avx)) {
        assert(!vDst.isEqualIfNotInherited(op));
        vmulps(vSrc, vSrc, op);
        vsubps(vDst, vSrc, vDst);
    } else {
        assert(!vDst.isEqualIfNotInherited(op));
        mulps(vSrc, op);
        subps(vSrc, vDst);
        movups(vDst, vSrc);
    }
}

void JitKernelBase::uni_vpaddd(const Xbyak::Ymm& vDst,
                               const Xbyak::Ymm& vSrc,
                               const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vpaddd(vDst, vSrc, op);
    } else if (isValidIsa(x64::avx)) {
        Xbyak::Xmm xmmDst(vDst.getIdx());
        vmovups(vDst, vSrc);
        if (op.isYMM()) {
            Xbyak::Ymm ymmOp(op.getIdx());
            Xbyak::Xmm xmmOp(op.getIdx());
            paddd(xmmDst, xmmOp);
            vperm2f128(vDst, vDst, vDst, 0x1);
            vperm2f128(ymmOp, ymmOp, ymmOp, 0x1);
            paddd(xmmDst, xmmOp);
            vperm2f128(vDst, vDst, vDst, 0x1);
            vperm2f128(ymmOp, ymmOp, ymmOp, 0x1);
        } else if (op.isMEM()) {
            const int vlen = x64::cpu_isa_traits<x64::sse41>::vlen;
            paddd(xmmDst, op.getAddress());
            vperm2f128(vDst, vDst, vDst, 0x1);
            paddd(xmmDst, ptr[op.getAddress().getRegExp() + vlen]);
            vperm2f128(vDst, vDst, vDst, 0x1);
        } else {
            IE_THROW() << "Not supported operand type.";
        }
    } else if (isValidIsa(x64::sse41)) {
        assert(vDst.getIdx() != vSrc.getIdx());
        paddd(vDst, op);
    } else {
        IE_THROW() << "Not defined behavior for instruction 'vpaddd' in current instructions set.";
    }
}

void JitKernelBase::uni_vpsubd(const Xbyak::Ymm& vDst,
                               const Xbyak::Ymm& vSrc,
                               const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vpsubd(vDst, vSrc, op);
    } else if (isValidIsa(x64::avx)) {
        Xbyak::Xmm xmmDst(vDst.getIdx());
        vmovups(vDst, vSrc);
        if (op.isYMM()) {
            Xbyak::Ymm ymmOp(op.getIdx());
            Xbyak::Xmm xmmOp(op.getIdx());
            psubd(xmmDst, xmmOp);
            vperm2f128(vDst, vDst, vDst, 0x1);
            vperm2f128(ymmOp, ymmOp, ymmOp, 0x1);
            psubd(xmmDst, xmmOp);
            vperm2f128(vDst, vDst, vDst, 0x1);
            vperm2f128(ymmOp, ymmOp, ymmOp, 0x1);
        } else if (op.isMEM()) {
            const int vlen = x64::cpu_isa_traits<x64::sse41>::vlen;
            psubd(xmmDst, op.getAddress());
            vperm2f128(vDst, vDst, vDst, 0x1);
            psubd(xmmDst, ptr[op.getAddress().getRegExp() + vlen]);
            vperm2f128(vDst, vDst, vDst, 0x1);
        } else {
            IE_THROW() << "Not supported operand type.";
        }
    } else if (isValidIsa(x64::sse41)) {
        assert(vDst.getIdx() != vSrc.getIdx());
        psubd(vDst, op);
    } else {
        IE_THROW() << "Not defined behavior for instruction 'vpsubd' in current instructions set.";
    }
}

void JitKernelBase::uni_vdivps(const Xbyak::Xmm& vDst,
                               const Xbyak::Operand& op1,
                               const Xbyak::Operand& op2) {
    if (isValidIsa(x64::avx)) {
        vdivps(vDst, op1, op2);
    } else {
        if (!vDst.isEqualIfNotInherited(op1)) {
            movups(vDst, op1);
        }
        divps(vDst, op2);
    }
}

void JitKernelBase::uni_vandps(const Xbyak::Xmm& vDst,
                               const Xbyak::Xmm& vSrs,
                               const Xbyak::Operand &op) {
    if (isValidIsa(x64::avx)) {
        vandps(vDst, vSrs, op);
    } else {
        if (!vDst.isEqualIfNotInherited(vSrs)) {
            movups(vDst, vSrs);
        }
        andps(vDst, op);
    }
}

void JitKernelBase::uni_vandnps(const Xbyak::Xmm& vDst,
                                const Xbyak::Xmm& vSrs,
                                const Xbyak::Operand &op) {
    if (isValidIsa(x64::avx)) {
        vandnps(vDst, vSrs, op);
    } else {
        if (!vDst.isEqualIfNotInherited(vSrs)) {
            movups(vDst, vSrs);
        }
        andnps(vDst, op);
    }
}

void JitKernelBase::gatherdd(const Xbyak::Xmm&    vDst,
                             const Xbyak::Reg64&  rSrcPtr,
                             const Xbyak::Xmm&    vSrcShift,
                             const Xbyak::Opmask& kReadMask,
                             const bool useMask,
                             const bool zeroFill) {
    if (kReadMask.getIdx() == 0) {
        IE_THROW() << "The vpgatherdd instruction cannot use the register k0 as mask.";
    }
    if (!useMask)
        kxnord(kReadMask, kReadMask, kReadMask);
    if (zeroFill)
        uni_vpxor(vDst, vDst, vDst);

    vpgatherdd(vDst | kReadMask, ptr[rSrcPtr + vSrcShift]);
}

void JitKernelBase::gatherdd(const Xbyak::Xmm&   vDst,
                             const Xbyak::Reg64& rSrcPtr,
                             const Xbyak::Xmm&   vSrcShift,
                             const Xbyak::Xmm&   vReadMask,
                             const bool useMask,
                             const bool zeroFill) {
    if (vDst.getIdx() == vSrcShift.getIdx() || vDst.getIdx() == vReadMask.getIdx() || vSrcShift.getIdx() == vReadMask.getIdx()) {
        IE_THROW() << "Any pair of the index, mask, or destination registers cannot be the same.";
    }
    if (zeroFill)
        pxor(vDst, vDst); // Don't use vpxor. It zeros the rest of the YMM register.

    if (isValidIsa(x64::avx2)) {
        if (!useMask)
            uni_vpcmpeqd(vReadMask, vReadMask, vReadMask);

        vpgatherdd(vDst, ptr[rSrcPtr + vSrcShift], vReadMask);
    } else {
        auto rAux = getReg64();
        Xbyak::Reg32 r32Aux = Xbyak::Reg32(rAux.getIdx());
        const uint8_t elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / sizeof(int);

        for (uint8_t i = 0; i < elPerVec; i++) {
            Xbyak::Label lLoopNext;
            if (useMask) {
                uni_vpextrd(r32Aux, vReadMask, i);
                cmp(r32Aux, 0); // TODO: check significant bit
                je(lLoopNext, T_NEAR);
            }
            uni_vpextrd(r32Aux, vSrcShift, i);
            pinsrd(vDst, ptr[rSrcPtr + rAux], i);

            if (useMask)
                L(lLoopNext);
        }
    }
}

void JitKernelBase::gatherdd(const Xbyak::Ymm&   vDst,
                             const Xbyak::Reg64& rSrcPtr,
                             const Xbyak::Ymm&   vSrcShift,
                             const Xbyak::Ymm&   vReadMask,
                             const bool useMask,
                             const bool zeroFill) {
    if (vDst.getIdx() == vSrcShift.getIdx() || vDst.getIdx() == vReadMask.getIdx() || vSrcShift.getIdx() == vReadMask.getIdx()) {
        IE_THROW() << "Any pair of the index, mask, or destination registers cannot be the same.";
    }
    if (isValidIsa(x64::avx2)) {
        if (!useMask)
            uni_vpcmpeqd(vReadMask, vReadMask, vReadMask);
        if (zeroFill)
            uni_vpxor(vDst, vDst, vDst);

        vpgatherdd(vDst, ptr[rSrcPtr + vSrcShift], vReadMask);
    } else {
        Xbyak::Xmm xmmDst      = Xbyak::Xmm(vDst.getIdx()),
                   xmmSrcShft  = Xbyak::Xmm(vSrcShift.getIdx()),
                   xmmReadMask = Xbyak::Xmm(vReadMask.getIdx());
        for (uint8_t i = 0; i < 2; i++) {
            gatherdd(xmmDst, rSrcPtr, xmmSrcShft, xmmReadMask, useMask, zeroFill);

            vperm2f128(vDst, vDst, vDst, 0x1);
            vperm2f128(vSrcShift, vSrcShift, vSrcShift, 0x1);
            if (useMask)
                vperm2f128(vReadMask, vReadMask, vReadMask, 0x1);
        }
    }
}

void JitKernelBase::uni_vpbroadcastd(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
    if (isValidIsa(x64::avx2)) {
        vpbroadcastd(x, op);
    } else if (isValidIsa(x64::avx)) {
        if (op.isMEM()) {
            vbroadcastss(x, op.getAddress());
        } else {
            vmovss(x, x, op);
            vpshufd(x, x, 0x0);
        }
    } else {
        movss(x, op);
        pshufd(x, x, 0x0);
    }
}

void JitKernelBase::uni_vpbroadcastd(const Xbyak::Ymm &x, const Xbyak::Operand &op) {
    if (isValidIsa(x64::avx2)) {
        vpbroadcastd(x, op);
    } else {
        if (op.isMEM()) {
            vbroadcastss(x, op.getAddress());
        } else {
            const Xbyak::Xmm t(x.getIdx());
            if (!t.isEqualIfNotInherited(op)) {
                vmovss(t, t, op);
            }
            vinsertf128(x, x, t, 1);
            vshufps(x, x, x, 0);
        }
    }
}

void JitKernelBase::fillRestWorkMask(const Xbyak::Opmask& dstMask,
                                     const Xbyak::Reg64& rWorkRest) {
    auto rOnes = getReg64();

    mov(rOnes, 0xFFFFFFFFFFFFFFFF);
    shlx(rOnes, rOnes, rWorkRest);
    not_(rOnes);
    kmovq(dstMask, rOnes);
}

void JitKernelBase::fillRestWorkMask(const Xbyak::Xmm& xmmDstMask,
                                     const Xbyak::Reg64& rWorkRest,
                                     const uint64_t typeSize) {
    if (!one_of(typeSize, 1, 2, 4, 8)) {
        IE_THROW() << "Could not fill data with type size " << typeSize;
    }
    Xbyak::Label lEnd;
    auto r32Ones = getReg32();
    Xbyak::Reg64 r64Ones(r32Ones.getIdx());
    auto elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;

    mov(r64Ones, 0xFFFFFFFFFFFFFFFF);
    for (uint8_t i = 0; i < elPerVec; i++) {
        cmp(rWorkRest, i);
        jle(lEnd, T_NEAR);

        if (typeSize == 1) {
            pinsrb(xmmDstMask, r32Ones, i);
        } else if (typeSize == 2) {
            pinsrw(xmmDstMask, r32Ones, i);
        } else if (typeSize == 4) {
            pinsrd(xmmDstMask, r32Ones, i);
        } else if (typeSize == 8) {
            pinsrq(xmmDstMask, r64Ones, i);
        }
    }
    L(lEnd);
}

void JitKernelBase::fillRestWorkMask(const Xbyak::Ymm& ymmDstMask,
                                     const Xbyak::Reg64& rWorkRest,
                                     const uint64_t typeSize) {
    if (!one_of(typeSize, 1, 2, 4, 8)) {
        IE_THROW() << "Could not fill data with type size " << typeSize;
    }
    Xbyak::Label lEnd;
    auto elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;
    auto r32Ones = getReg32();
    Xbyak::Reg64 r64Ones(r32Ones.getIdx());
    Xbyak::Xmm xmmDstMask(ymmDstMask.getIdx());

    mov(r64Ones, 0xFFFFFFFFFFFFFFFF);
    uni_vpxor(ymmDstMask, ymmDstMask, ymmDstMask);
    for (uint8_t i = 0; i < 2; i++) {
        Xbyak::Label lPerm;
        for (uint8_t j = 0; j < elPerVec; j++) {
            cmp(rWorkRest, i * elPerVec + j);
            jle(i == 0 ? lEnd : lPerm, T_NEAR);

            if (typeSize == 1) {
                pinsrb(xmmDstMask, r32Ones, j);
            } else if (typeSize == 2) {
                pinsrw(xmmDstMask, r32Ones, j);
            } else if (typeSize == 4) {
                pinsrd(xmmDstMask, r32Ones, j);
            } else if (typeSize == 8) {
                pinsrq(xmmDstMask, r64Ones, j);
            }
        }
        cmp(rWorkRest, elPerVec);
        je(lEnd, T_NEAR);
        L(lPerm);
        vperm2f128(ymmDstMask, ymmDstMask, ymmDstMask, 0x1);
    }
    L(lEnd);
}

void JitKernelBase::load(const Xbyak::Xmm&     vDst,
                         const Xbyak::Address& srcAddr,
                         const Xbyak::Reg64&   rLoadNum,
                         const size_t          typeSize,
                         const bool            zeroFilling) {
    if (!one_of(typeSize, 1, 2, 4, 8)) {
        IE_THROW() << "Could not load data with type size " << typeSize;
    }
    const uint8_t elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;
    Xbyak::Label lEnd;
    if (zeroFilling)
        pxor(vDst, vDst);

    for (uint8_t i = 0; i < elPerVec; i++) {
        cmp(rLoadNum, i);
        jle(lEnd, T_NEAR);

        const size_t offset = i * typeSize;
        if (typeSize == 1)
            pinsrb(vDst, ptr[srcAddr.getRegExp() + offset], i);
        else if (typeSize == 2)
            pinsrw(vDst, ptr[srcAddr.getRegExp() + offset], i);
        else if (typeSize == 4)
            pinsrd(vDst, ptr[srcAddr.getRegExp() + offset], i);
        else if (typeSize == 8)
            pinsrq(vDst, ptr[srcAddr.getRegExp() + offset], i);
    }
    L(lEnd);
}

void JitKernelBase::load(const Xbyak::Ymm&     vDst,
                         const Xbyak::Address& srcAddr,
                         const Xbyak::Reg64&   rLoadNum,
                         const size_t          typeSize,
                         const bool            zeroFilling) {
    if (!one_of(typeSize, 1, 2, 4, 8)) {
        IE_THROW() << "Could not load data with type size " << typeSize;
    }
    const size_t elPerXmm = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;
    Xbyak::Label lEnd;
    if (zeroFilling)
        uni_vpxor(vDst, vDst, vDst);
    Xbyak::Xmm xmmDst(vDst.getIdx());

    for (size_t i = 0lu; i < 2lu; i++) {
        Xbyak::Label lPerm;
        const size_t idx = i * elPerXmm;
        const size_t offset0 = idx * typeSize;

        for (size_t j = 0lu; j < elPerXmm; j++) {
            cmp(rLoadNum, j + idx);
            jle(i == 0 ? lEnd : lPerm, T_NEAR);

            const size_t offset = offset0 + j * typeSize;
            if (typeSize == 1)
                pinsrb(xmmDst, ptr[srcAddr.getRegExp() + offset], j);
            else if (typeSize == 2)
                pinsrw(xmmDst, ptr[srcAddr.getRegExp() + offset], j);
            else if (typeSize == 4)
                pinsrd(xmmDst, ptr[srcAddr.getRegExp() + offset], j);
            else if (typeSize == 8)
                pinsrq(xmmDst, ptr[srcAddr.getRegExp() + offset], j);
        }

        L(lPerm);
        vperm2f128(vDst, vDst, vDst, 0x1);
    }
    L(lEnd);
}

void JitKernelBase::store(const Xbyak::Address& dstAddr,
                          const Xbyak::Xmm&     vSrc,
                          const Xbyak::Reg64&   rToStoreNum,
                          const size_t          typeSize) {
    if (!one_of(typeSize, 1, 2, 4, 8)) {
        IE_THROW() << "Could not store data with type size " << typeSize;
    }
    Xbyak::Label lEnd;
    const size_t elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;

    for (size_t i = 0; i < elPerVec; i++) {
        cmp(rToStoreNum, i);
        jle(lEnd, T_NEAR);

        const size_t offset = i * typeSize;
        if (typeSize == 1) {
            uni_vpextrb(ptr[dstAddr.getRegExp() + offset], vSrc, i);
        } else if (typeSize == 2) {
            uni_vpextrw(ptr[dstAddr.getRegExp() + offset], vSrc, i);
        } else if (typeSize == 4) {
            uni_vpextrd(ptr[dstAddr.getRegExp() + offset], vSrc, i);
        } else if (typeSize == 8) {
            uni_vpextrq(ptr[dstAddr.getRegExp() + offset], vSrc, i);
        }
    }
    L(lEnd);
}

void JitKernelBase::store(const Xbyak::Address& dstAddr,
                          const Xbyak::Ymm&     vSrc,
                          const Xbyak::Reg64&   rToStoreNum,
                          const size_t          typeSize) {
    if (!one_of(typeSize, 1, 2, 4, 8)) {
        IE_THROW() << "Could not store data with type size " << typeSize;
    }
    Xbyak::Label lEnd;
    Xbyak::Xmm xmmSrc(vSrc.getIdx());
    const size_t elPerXmm = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;

    for (int i = 0; i < 2; i++) {
        Xbyak::Label lPerm;
        const size_t idx = i * elPerXmm;
        const size_t offset0 = idx * typeSize;

        for (size_t j = 0; j < elPerXmm; j++) {
            cmp(rToStoreNum, j + idx);
            jle(i == 0 ? lEnd : lPerm, T_NEAR);

            const size_t offset = offset0 + j * typeSize;
            if (typeSize == 8) {
                uni_vpextrq(ptr[dstAddr.getRegExp() + offset], xmmSrc, j);
            } else if (typeSize == 4) {
                uni_vpextrd(ptr[dstAddr.getRegExp() + offset], xmmSrc, j);
            } else if (typeSize == 2) {
                uni_vpextrw(ptr[dstAddr.getRegExp() + offset], xmmSrc, j);
            } else if (typeSize == 1) {
                uni_vpextrb(ptr[dstAddr.getRegExp() + offset], xmmSrc, j);
            }
        }

        L(lPerm);
        vperm2f128(vSrc, vSrc, vSrc, 0x1);
    }
    L(lEnd);
}

void JitKernelBase::memMovDD(const Xbyak::Reg64& rDst,
                             const Xbyak::Reg64& rSrc,
                             const Xbyak::Xmm&   vReadMask,
                             const Xbyak::Xmm&   vSrcShift,
                             const Xbyak::Reg64& rToStoreNum,
                             const bool          useMask,
                             const bool          zeroFill) {
    Xbyak::Label lEnd;
    auto rAux = getReg64();
    Xbyak::Reg32 r32Aux = Xbyak::Reg32(rAux.getIdx());
    const uint8_t typeSize = sizeof(int);
    const uint8_t elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;

    for (uint8_t i = 0; i < elPerVec; i++) {
        cmp(rToStoreNum, i);
        jle(lEnd, T_NEAR);

        Xbyak::Label lLoopNext;
        if (useMask) {
            uni_vpextrd(r32Aux, vReadMask, i);
            cmp(r32Aux, 0);
            if (zeroFill) {
                Xbyak::Label lNotZero;
                jne(lNotZero, T_NEAR);
                mov(ptr[rDst.getReg() + i * typeSize], r32Aux);
                jmp(lLoopNext, T_NEAR);
                L(lNotZero);
            } else {
                je(lLoopNext, T_NEAR);
            }
        }
        uni_vpextrd(r32Aux, vSrcShift, i);
        mov(r32Aux, ptr[rSrc.getReg() + rAux]);
        mov(ptr[rDst.getReg() + i * typeSize], r32Aux);

        L(lLoopNext);
    }
    L(lEnd);
}

void JitKernelBase::memMovDD(const Xbyak::Reg64& rDst,
                             const Xbyak::Reg64& rSrc,
                             const Xbyak::Ymm&   vReadMask,
                             const Xbyak::Ymm&   vSrcShift,
                             const Xbyak::Reg64& rToStoreNum,
                             const bool          useMask,
                             const bool          zeroFill) {
    Xbyak::Label lEnd;
    if (isValidIsa(x64::avx2)) {
        auto vAux = RegistersPool::Reg<Xbyak::Ymm>(registersPool);
        gatherdd(vAux, rSrc, vSrcShift, vReadMask, useMask, zeroFill);
        store(ptr[rDst], vAux, rToStoreNum, sizeof(int));
    } else if (isValidIsa(x64::avx)) {
        const uint8_t typeSize = sizeof(int);
        const uint8_t elPerXmm = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;
        Xbyak::Xmm xmmReadMask  = Xbyak::Xmm(vReadMask.getIdx()),
                   xmmSrcShft   = Xbyak::Xmm(vSrcShift.getIdx());
        for (uint8_t i = 0; i < 2; i++) {
            memMovDD(rDst, rSrc, xmmReadMask, xmmSrcShft, rToStoreNum, useMask, zeroFill);

            if (i == 0) {
                cmp(rToStoreNum, elPerXmm);
                jle(lEnd, T_NEAR);
                sub(rToStoreNum, elPerXmm);
                add(rDst, typeSize * elPerXmm);
            } else {
                add(rToStoreNum, elPerXmm);
                sub(rDst, typeSize * elPerXmm);
            }

            vperm2f128(vSrcShift, vSrcShift, vSrcShift, 0x1);
            if (useMask)
                vperm2f128(vReadMask, vReadMask, vReadMask, 0x1);
        }
    }
    L(lEnd);
}
