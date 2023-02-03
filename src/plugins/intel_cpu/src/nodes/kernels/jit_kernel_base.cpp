// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel_base.hpp"

using namespace ov::intel_cpu::kernel;
using namespace dnnl::impl::cpu;
using namespace Xbyak;
using Precision = typename InferenceEngine::Precision;


JitKernelBase::JitKernelBase(const char* name) : x64::jit_generator(name) {
    if (isValidIsa(x64::avx512_core)) {
        vcvtneps2bf16.reset(new jit_uni_vcvtneps2bf16(this, x64::avx512_core));
    } else if (isValidIsa(x64::avx2)) {
        vcvtneps2bf16.reset(new jit_uni_vcvtneps2bf16(this, x64::avx2));
    } else if (isValidIsa(x64::sse41)) {
        vcvtneps2bf16.reset(new jit_uni_vcvtneps2bf16(this, x64::sse41));
    }
}

void JitKernelBase::uni_vfmsub132ps(const Xmm& vDst,
                                    const Xmm& vSrc,
                                    const Operand& op) {
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

void JitKernelBase::uni_vfnmadd132ps(const Xmm& vDst,
                                     const Xmm& vSrc,
                                     const Operand& op) {
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

void JitKernelBase::uni_vfmsub231ps(const Xmm& vDst,
                                    const Xmm& vSrc,
                                    const Operand& op) {
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

void JitKernelBase::uni_vpaddd(const Ymm& vDst,
                               const Ymm& vSrc,
                               const Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vpaddd(vDst, vSrc, op);
    } else if (isValidIsa(x64::avx)) {
        Xmm xmmDst(vDst.getIdx());
        vmovups(vDst, vSrc);
        if (op.isYMM()) {
            Ymm ymmOp(op.getIdx());
            Xmm xmmOp(op.getIdx());
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

void JitKernelBase::uni_vaddpd(const Xmm &vDst, const Operand &op1, const Operand &op2) {
    if (isValidIsa(x64::avx)) {
        vaddpd(vDst, op1, op2);
    } else {
        if (vDst.getIdx() != op1.getIdx()) {
            movupd(vDst, op1);
        }
        addpd(vDst, op2);
    }
}

void JitKernelBase::uni_vpsubd(const Ymm& vDst,
                               const Ymm& vSrc,
                               const Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vpsubd(vDst, vSrc, op);
    } else if (isValidIsa(x64::avx)) {
        Xmm xmmDst(vDst.getIdx());
        vmovups(vDst, vSrc);
        if (op.isYMM()) {
            Ymm ymmOp(op.getIdx());
            Xmm xmmOp(op.getIdx());
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
        if (vDst.getIdx() != vSrc.getIdx()) {
            movups(vDst, vSrc);
        }
        psubd(vDst, op);
    } else {
        IE_THROW() << "Not defined behavior for instruction 'vpsubd' in current instructions set.";
    }
}

void JitKernelBase::uni_vmulpd(const Xmm& vDst,
                               const Operand& op1,
                               const Operand& op2) {
    if (isValidIsa(x64::avx)) {
        vmulpd(vDst, op1, op2);
    } else {
        if (vDst.getIdx() != op1.getIdx()) {
            movupd(vDst, op1);
        }
        mulpd(vDst, op2);
    }
}

void JitKernelBase::uni_vdivps(const Xmm& vDst,
                               const Operand& op1,
                               const Operand& op2) {
    if (isValidIsa(x64::avx)) {
        vdivps(vDst, op1, op2);
    } else {
        if (!vDst.isEqualIfNotInherited(op1)) {
            movups(vDst, op1);
        }
        divps(vDst, op2);
    }
}

void JitKernelBase::uni_vdivpd(const Xmm& vDst,
                               const Operand& op1,
                               const Operand& op2) {
    if (isValidIsa(x64::avx)) {
        vdivpd(vDst, op1, op2);
    } else {
        if (vDst.getIdx() != op1.getIdx()) {
            movupd(vDst, op1);
        }
        divpd(vDst, op2);
    }
}

void JitKernelBase::uni_vsqrtpd(const Xmm &vDst, const Operand &op) {
    if (isValidIsa(x64::avx)) {
        vsqrtpd(vDst, op);
    } else {
        sqrtpd(vDst, op);
    }
}

void JitKernelBase::uni_vandps(const Xmm& vDst,
                               const Xmm& vSrs,
                               const Operand &op) {
    if (isValidIsa(x64::avx)) {
        vandps(vDst, vSrs, op);
    } else {
        if (vDst.getIdx() != vSrs.getIdx()) {
            movups(vDst, vSrs);
        }
        andps(vDst, op);
    }
}

void JitKernelBase::uni_vandpd(const Xmm& vDst,
                               const Xmm& vSrs,
                               const Operand &op) {
    if (isValidIsa(x64::avx)) {
        vandpd(vDst, vSrs, op);
    } else {
        if (vDst.getIdx() != vSrs.getIdx()) {
            movupd(vDst, vSrs);
        }
        andpd(vDst, op);
    }
}

void JitKernelBase::uni_vandnps(const Xmm& vDst,
                                const Xmm& vSrs,
                                const Operand &op) {
    if (isValidIsa(x64::avx)) {
        vandnps(vDst, vSrs, op);
    } else {
        if (!vDst.isEqualIfNotInherited(vSrs)) {
            movups(vDst, vSrs);
        }
        andnps(vDst, op);
    }
}

void JitKernelBase::uni_vorpd(const Xmm& vDst,
                              const Xmm& vSrs,
                              const Operand &op) {
    if (isValidIsa(x64::avx)) {
        vorpd(vDst, vSrs, op);
    } else {
        if (vDst.getIdx() != vSrs.getIdx()) {
            movupd(vDst, vSrs);
        }
        orpd(vDst, op);
    }
}

void JitKernelBase::uni_vcmppd(const Xmm &vDst,
                               const Xmm &vSrs,
                               const Operand &op,
                               const uint8_t imm) {
    if (isValidIsa(x64::avx)) {
        vcmppd(vDst, vSrs, op, imm);
    } else {
        if (vDst.getIdx() != vSrs.getIdx()) {
            movupd(vDst, vSrs);
        }
        cmppd(vDst, op, imm);
    }
}

void JitKernelBase::uni_vmaxpd(const Xmm &vDst, const Operand &op1, const Operand &op2) {
    if (isValidIsa(x64::avx)) {
        vmaxpd(vDst, op1, op2);
    } else {
        if (vDst.getIdx() != op1.getIdx()) {
            movupd(vDst, op1);
        }
        maxpd(vDst, op2);
    }
}

void JitKernelBase::uni_vminpd(const Xmm &vDst, const Operand &op1, const Operand &op2) {
    if (isValidIsa(x64::avx)) {
        vminpd(vDst, op1, op2);
    } else {
        if (vDst.getIdx() != op1.getIdx()) {
            movupd(vDst, op1);
        }
        minpd(vDst, op2);
    }
}

void JitKernelBase::uni_vcvtpd2dq(const Xbyak::Xmm &vDst, const Xbyak::Operand &op) {
    if (isValidIsa(x64::avx)) {
        vcvtpd2dq(vDst, op);
    } else {
        cvtpd2dq(vDst, op);
    }
}

void JitKernelBase::uni_vcvtpd2ps(const Xbyak::Xmm &vDst, const Xbyak::Operand &op) {
    if (isValidIsa(x64::avx)) {
        vcvtpd2ps(vDst, op);
    } else {
        cvtpd2ps(vDst, op);
    }
}

void JitKernelBase::gatherdd(const Xmm&    vDst,
                             const Reg64&  rSrcPtr,
                             const Xmm&    vSrcShift,
                             const Opmask& kReadMask,
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

void JitKernelBase::gatherdd(const Xmm&   vDst,
                             const Reg64& rSrcPtr,
                             const Xmm&   vSrcShift,
                             const Xmm&   vReadMask,
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
        Reg32 r32Aux = Reg32(rAux.getIdx());
        const uint8_t elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / sizeof(int);

        for (uint8_t i = 0; i < elPerVec; i++) {
            Label lLoopNext;
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

void JitKernelBase::gatherdd(const Ymm&   vDst,
                             const Reg64& rSrcPtr,
                             const Ymm&   vSrcShift,
                             const Ymm&   vReadMask,
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
        Xmm xmmDst      = Xmm(vDst.getIdx()),
                   xmmSrcShft  = Xmm(vSrcShift.getIdx()),
                   xmmReadMask = Xmm(vReadMask.getIdx());
        for (uint8_t i = 0; i < 2; i++) {
            gatherdd(xmmDst, rSrcPtr, xmmSrcShft, xmmReadMask, useMask, zeroFill);

            vperm2f128(vDst, vDst, vDst, 0x1);
            vperm2f128(vSrcShift, vSrcShift, vSrcShift, 0x1);
            if (useMask)
                vperm2f128(vReadMask, vReadMask, vReadMask, 0x1);
        }
    }
}

void JitKernelBase::uni_vpbroadcastd(const Xmm &x, const Operand &op) {
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

void JitKernelBase::uni_vpbroadcastd(const Ymm &x, const Operand &op) {
    if (isValidIsa(x64::avx2)) {
        vpbroadcastd(x, op);
    } else {
        if (op.isMEM()) {
            vbroadcastss(x, op.getAddress());
        } else {
            const Xmm t(x.getIdx());
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

void JitKernelBase::load(const Xmm&     vDst,
                         const Address& srcAddr,
                         const Reg64&   rLoadNum,
                         const size_t   typeSize,
                         const bool     zeroFilling) {
    if (!one_of(typeSize, 1, 2, 4, 8)) {
        IE_THROW() << "Could not load data with type size " << typeSize;
    }
    const uint8_t elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;
    Label lEnd;
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

void JitKernelBase::load(const Ymm&     vDst,
                         const Address& srcAddr,
                         const Reg64&   rLoadNum,
                         const size_t   typeSize,
                         const bool     zeroFilling) {
    if (!one_of(typeSize, 1, 2, 4, 8)) {
        IE_THROW() << "Could not load data with type size " << typeSize;
    }
    const size_t elPerXmm = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;
    Label lEnd;
    if (zeroFilling)
        uni_vpxor(vDst, vDst, vDst);
    Xmm xmmDst(vDst.getIdx());

    for (size_t i = 0lu; i < 2lu; i++) {
        Label lPerm;
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

void JitKernelBase::store(const Address& dstAddr,
                          const Xmm&     vSrc,
                          const Reg64&   rToStoreNum,
                          const size_t          typeSize) {
    if (!one_of(typeSize, 1, 2, 4, 8)) {
        IE_THROW() << "Could not store data with type size " << typeSize;
    }
    Label lEnd;
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

void JitKernelBase::store(const Address& dstAddr,
                          const Ymm&     vSrc,
                          const Reg64&   rToStoreNum,
                          const size_t          typeSize) {
    if (!one_of(typeSize, 1, 2, 4, 8)) {
        IE_THROW() << "Could not store data with type size " << typeSize;
    }
    Label lEnd;
    Xmm xmmSrc(vSrc.getIdx());
    const size_t elPerXmm = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;

    for (int i = 0; i < 2; i++) {
        Label lPerm;
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

void JitKernelBase::memMovDD(const Reg64& rDst,
                             const Reg64& rSrc,
                             const Xmm&   vReadMask,
                             const Xmm&   vSrcShift,
                             const Reg64& rToStoreNum,
                             const bool          useMask,
                             const bool          zeroFill) {
    Label lEnd;
    auto rAux = getReg64();
    Reg32 r32Aux = Reg32(rAux.getIdx());
    const uint8_t typeSize = sizeof(int);
    const uint8_t elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;

    for (uint8_t i = 0; i < elPerVec; i++) {
        cmp(rToStoreNum, i);
        jle(lEnd, T_NEAR);

        Label lLoopNext;
        if (useMask) {
            uni_vpextrd(r32Aux, vReadMask, i);
            cmp(r32Aux, 0);
            if (zeroFill) {
                Label lNotZero;
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

void JitKernelBase::memMovDD(const Reg64& rDst,
                             const Reg64& rSrc,
                             const Ymm&   vReadMask,
                             const Ymm&   vSrcShift,
                             const Reg64& rToStoreNum,
                             const bool          useMask,
                             const bool          zeroFill) {
    Label lEnd;
    if (isValidIsa(x64::avx2)) {
        auto vAux = RegistersPool::Reg<Ymm>(registersPool);
        gatherdd(vAux, rSrc, vSrcShift, vReadMask, useMask, zeroFill);
        store(ptr[rDst], vAux, rToStoreNum, sizeof(int));
    } else if (isValidIsa(x64::avx)) {
        const uint8_t typeSize = sizeof(int);
        const uint8_t elPerXmm = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;
        Xmm xmmReadMask  = Xmm(vReadMask.getIdx()),
                   xmmSrcShft   = Xmm(vSrcShift.getIdx());
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

void JitKernelBase::loadVector(const Xmm &vDst,
                               const Address &srcAdr,
                               const InferenceEngine::Precision &dstPrc,
                               const InferenceEngine::Precision &srcPrc) {
    Xmm xmmDst = Xmm(vDst.getIdx());
    Ymm ymmDst = Ymm(vDst.getIdx());

    switch (srcPrc) {
        case Precision::FP64:
            if (x64::mayiuse(x64::avx512_core) && one_of(dstPrc, Precision::I64, Precision::I32, Precision::FP32)) {
                if (dstPrc == Precision::I64) {
                    vcvtpd2qq(vDst, srcAdr);
                } else if (dstPrc == Precision::I32) {
                    uni_vcvtpd2dq(vDst.isZMM() ? ymmDst : vDst, srcAdr);
                } else if (dstPrc == Precision::FP32) {
                    uni_vcvtpd2ps(vDst.isZMM() ? ymmDst : vDst, srcAdr);
                }
            } else if (!x64::mayiuse(x64::avx512_core) && one_of(dstPrc, Precision::FP32, Precision::I32)) {
                if (dstPrc == Precision::FP32) {
                    uni_vcvtpd2ps(xmmDst, srcAdr);
                } else if (dstPrc == Precision::I32) {
                    uni_vcvtpd2dq(xmmDst, srcAdr);
                }
            } else {
                uni_vmovups(vDst, srcAdr);
            }
            break;
        case Precision::I64:
            if (x64::mayiuse(x64::avx512_core) && one_of(dstPrc, Precision::FP64, Precision::FP32)) {
                if (dstPrc == Precision::FP64) {
                    vcvtqq2pd(vDst, srcAdr);
                } else if (dstPrc == Precision::FP32) {
                    vcvtqq2ps(vDst.isZMM() ? ymmDst : vDst, srcAdr);
                }
            } else {
                uni_vmovups(vDst, srcAdr);
            }
            break;
        case Precision::FP32:
            if (dstPrc == Precision::I32) {
                uni_vcvtps2dq(vDst, srcAdr);
            } else {
                uni_vmovups(vDst, srcAdr);
            }
            break;
        case Precision::I32:
            if (dstPrc == Precision::FP64) {
                uni_vcvtdq2pd(vDst, srcAdr);
            } else if (dstPrc == Precision::FP32) {
                uni_vcvtdq2ps(vDst, srcAdr);
            } else {
                uni_vmovups(vDst, srcAdr);
            }
            break;
        case Precision::BF16:
            uni_vpmovzxwd(vDst, srcAdr);
            uni_vpslld(vDst, vDst, 16);
            break;
        case Precision::U16:
            if (one_of(dstPrc, Precision::FP32, Precision::I32)) {
                uni_vpmovzxwd(vDst, srcAdr);
            } else {
                uni_vmovups(vDst, srcAdr);
            }
            break;
        case Precision::I16:
            if (one_of(dstPrc, Precision::FP32, Precision::I32)) {
                uni_vpmovsxwd(vDst, srcAdr);
            } else {
                uni_vmovups(vDst, srcAdr);
            }
            break;
        case Precision::I8:
            if (one_of(dstPrc, Precision::FP32, Precision::I32)) {
                uni_vpmovsxbd(vDst, srcAdr);
            } else {
                uni_vmovups(vDst, srcAdr);
            }
            break;
        case Precision::U8:
            if (one_of(dstPrc, Precision::FP32, Precision::I32)) {
                uni_vpmovzxbd(vDst, srcAdr);
            } else {
                uni_vmovups(vDst, srcAdr);
            }
            break;
        default:
            IE_THROW() << "Unsupported source precision: " << srcPrc;
    }

    switch (dstPrc) {
        case Precision::FP32:
            if (!x64::mayiuse(x64::avx512_core) && (srcPrc == Precision::I64)) {
                // Do conversion later.
            }
            if (one_of(srcPrc, Precision::U8, Precision::I8, Precision::I16, Precision::U16)) {
                uni_vcvtdq2ps(vDst, vDst);
            }
            break;
        case Precision::I32:
            if (x64::mayiuse(x64::avx512_core)) {
                if (srcPrc == Precision::I64) {
                    vpmovsqd(vDst, vDst);
                }
            } else {
                if (srcPrc == Precision::I64) {
                    // Do conversion later.
                }
            }
            if (one_of(srcPrc, Precision::BF16)) {
                uni_vcvtps2dq(vDst, vDst);
            }
            break;
        case Precision::I64:
        case Precision::FP64:
            break;
        default:
            IE_THROW() << "Unsupported destination precision: " << dstPrc;
    }
}

void JitKernelBase::loadScalar(const Xmm &vDst,
                               const Address &srcAdr,
                               const InferenceEngine::Precision &dstPrc,
                               const InferenceEngine::Precision &srcPrc) {
    Address srcAdrBcst(srcAdr.getBit(), true, srcAdr.getRegExp());

    switch (srcPrc) {
        case Precision::FP64:
            if (x64::mayiuse(x64::avx512_core) && one_of(dstPrc, Precision::I64, Precision::I32, Precision::FP32)) {
                if (dstPrc == Precision::I64) {
                    vcvtpd2qq(vDst, srcAdrBcst);
                } else if (dstPrc == Precision::I32) {
                    vcvtpd2dq(vDst, srcAdrBcst);
                } else if (dstPrc == Precision::FP32) {
                    vcvtpd2ps(vDst, srcAdrBcst);
                }
            } else {
                uni_vmovsd(vDst, srcAdr);
            }
            break;
        case Precision::I64:
            if (x64::mayiuse(x64::avx512_core) && one_of(dstPrc, Precision::FP64, Precision::FP32)) {
                if (dstPrc == Precision::FP64) {
                    vcvtqq2pd(vDst, srcAdrBcst);
                } else if (dstPrc == Precision::FP32) {
                    vcvtqq2ps(vDst, srcAdrBcst);
                }
            } else {
                uni_vmovsd(vDst, srcAdr);
            }
            break;
        case Precision::FP32:
            if (x64::mayiuse(x64::avx512_core) && one_of(dstPrc, Precision::FP64, Precision::I32)) {
                if (dstPrc == Precision::FP64) {
                    vcvtps2pd(vDst, srcAdrBcst);
                } else if (dstPrc == Precision::I32) {
                    vcvtps2dq(vDst, srcAdrBcst);
                }
            } else {
                uni_vmovss(vDst, srcAdr);
            }
            break;
        case Precision::I32:
            if (x64::mayiuse(x64::avx512_core) && one_of(dstPrc, Precision::FP32, Precision::FP64)) {
                if (dstPrc == Precision::FP32) {
                    vcvtdq2ps(vDst, srcAdrBcst);
                } else if (dstPrc == Precision::FP64) {
                    vcvtdq2pd(vDst, srcAdrBcst);
                }
            } else {
                uni_vmovss(vDst, srcAdr);
            }
            break;
        case Precision::BF16:
            uni_vpinsrw(vDst, vDst, srcAdr, 0);
            uni_vpslld(vDst, vDst, 16);
            break;
        case Precision::I16:
            uni_vpinsrw(vDst, vDst, srcAdr, 0);
            uni_vpmovsxwd(vDst, srcAdr);
            break;
        case Precision::U16:
            uni_vpinsrw(vDst, vDst, srcAdr, 0);
            uni_vpmovzxwd(vDst, srcAdr);
            break;
        case Precision::I8:
            pinsrb(vDst, srcAdr, 0);
            uni_vpmovsxbd(vDst, vDst);
            break;
        case Precision::U8:
            pinsrb(vDst, srcAdr, 0);
            uni_vpmovzxbd(vDst, vDst);
            break;
        default:
            IE_THROW() << "Unsupported source precision: " << srcPrc;
    }

    switch (dstPrc) {
        case Precision::FP32:
            if (x64::mayiuse(x64::avx512_core)) {
                if (one_of(srcPrc, Precision::U8, Precision::I8, Precision::U16, Precision::I16)) {
                    uni_vcvtdq2ps(vDst, vDst);
                }
            } else {
                if (srcPrc == Precision::FP64) {
                    uni_vcvtpd2ps(vDst, vDst);
                } else if (srcPrc == Precision::I64) {
                    // Do conversion later.
                } else if (one_of(srcPrc, Precision::U8, Precision::I8, Precision::U16, Precision::I16, Precision::I32)) {
                    uni_vcvtdq2ps(vDst, vDst);
                }
            }
            break;
        case Precision::I32:
            if (!x64::mayiuse(x64::avx512_core)) {
                if (srcPrc == Precision::I64) {
                    // Do conversion later.
                } else if (one_of(srcPrc, Precision::FP32, Precision::BF16)) {
                    uni_vcvtps2dq(vDst, vDst);
                }
            } else if (srcPrc == Precision::I64) {
                vpmovsqd(vDst, vDst);
            }
            break;
        case Precision::I64:
        case Precision::FP64:
            break;
        default:
            IE_THROW() << "Unsupported destination precision: " << dstPrc;
    }
}

void JitKernelBase::loadWithBcst(const Xmm &vDst,
                                 const Address &srcAdr,
                                 const InferenceEngine::Precision &dstPrc,
                                 const InferenceEngine::Precision &srcPrc) {
    Address srcAdrBcst(srcAdr.getBit(), true, srcAdr.getRegExp());
    auto xmmDst = Xmm(vDst.getIdx());

    switch (srcPrc) {
        case Precision::FP64:
            if (x64::mayiuse(x64::avx512_core) && one_of(dstPrc, Precision::I64, Precision::I32, Precision::FP32)) {
                if (dstPrc == Precision::I64) {
                    vcvtpd2qq(vDst, srcAdrBcst);
                } else if (dstPrc == Precision::I32) {
                    vcvtpd2dq(vDst, srcAdrBcst);
                } else if (dstPrc == Precision::FP32) {
                    vcvtpd2ps(vDst, srcAdrBcst);
                }
            } else {
                uni_vbroadcastsd(vDst, srcAdr); // does not work with XMM, use vpbroadcastq instead
            }
            break;
        case Precision::I64:
            if (x64::mayiuse(x64::avx512_core) && one_of(dstPrc, Precision::FP64, Precision::FP32)) {
                if (dstPrc == Precision::FP64) {
                    vcvtqq2pd(vDst, srcAdrBcst);
                } else if (dstPrc == Precision::FP32) {
                    vcvtqq2ps(vDst, srcAdrBcst);
                }
            } else {
                uni_vbroadcastsd(vDst, srcAdr);
            }
            break;
        case Precision::FP32:
            if (x64::mayiuse(x64::avx512_core) && one_of(dstPrc, Precision::FP64, Precision::I32)) {
                if (dstPrc == Precision::FP64) {
                    vcvtps2pd(vDst, srcAdrBcst);
                } else if (dstPrc == Precision::I32) {
                    vcvtps2dq(vDst, srcAdrBcst);
                }
            } else {
                uni_vbroadcastss(vDst, srcAdr);
            }
            break;
        case Precision::I32:
            if (x64::mayiuse(x64::avx512_core) && one_of(dstPrc, Precision::FP32, Precision::FP64)) {
                if (dstPrc == Precision::FP32) {
                    vcvtdq2ps(vDst, srcAdrBcst);
                } else if (dstPrc == Precision::FP64) {
                    vcvtdq2pd(vDst, srcAdrBcst);
                }
            } else {
                uni_vbroadcastss(vDst, srcAdr);
            }
            break;
        case Precision::BF16:
            uni_vpinsrw(vDst, vDst, srcAdr, 0);
            uni_vpslld(vDst, vDst, 16);
            break;
        case Precision::I16:
            uni_vpinsrw(vDst, vDst, srcAdr, 0);
            uni_vpmovsxwd(vDst, srcAdr);
            break;
        case Precision::U16:
            uni_vpinsrw(vDst, vDst, srcAdr, 0);
            uni_vpmovzxwd(vDst, srcAdr);
            break;
        case Precision::I8:
            if (dstPrc == Precision::I32) {
                pinsrb(vDst, srcAdr, 0);
                uni_vpmovsxbd(vDst, vDst);
            } else {
                vpbroadcastb(vDst, srcAdr);
            }
            break;
        case Precision::U8:
            if (dstPrc == Precision::I32) {
                pinsrb(vDst, srcAdr, 0);
                uni_vpmovzxbd(vDst, vDst);
            } else {
                vpbroadcastb(vDst, srcAdr);
            }
            break;
        default:
            IE_THROW() << "Unsupported source precision: " << srcPrc;
    }

    switch (dstPrc) {
        case Precision::FP32:
            if (x64::mayiuse(x64::avx512_core)) {
                if (one_of(srcPrc, Precision::U8, Precision::I8, Precision::U16, Precision::I16)) {
                    uni_vcvtdq2ps(vDst, vDst);
                }
            } else {
                if (srcPrc == Precision::FP64) {
                    uni_vcvtpd2ps(vDst, vDst);
                } else if (srcPrc == Precision::I64) {
                    // Do conversion later.
                } else if (one_of(srcPrc, Precision::U8, Precision::I8, Precision::U16, Precision::I16, Precision::I32)) {
                    uni_vcvtdq2ps(vDst, vDst);
                }
            }
            break;
        case Precision::I32:
            if (!x64::mayiuse(x64::avx512_core)) {
                if (srcPrc == Precision::I64) {
                    // Do conversion later.
                } else if (one_of(srcPrc, Precision::FP32, Precision::BF16)) {
                    uni_vcvtps2dq(vDst, vDst);
                }
            } else if (srcPrc == Precision::I64) {
                vpmovsqd(vDst, vDst);
            }
            break;
        case Precision::I64:
        case Precision::FP64:
            break;
        default:
            IE_THROW() << "Unsupported destination precision: " << dstPrc;
    }
}

void JitKernelBase::storeVector(const Address &dstAdr,
                                const Xmm &vSrc,
                                const InferenceEngine::Precision &dstPrc,
                                const InferenceEngine::Precision &srcPrc) {
    auto xmmSrc = Xmm(vSrc.getIdx());
    auto ymmSrc = Ymm(vSrc.getIdx());

    switch (srcPrc) {
        case Precision::FP64:
            if (dstPrc == Precision::FP32) {
                uni_vcvtpd2ps(x64::mayiuse(x64::avx512_core) ? ymmSrc : xmmSrc, vSrc);
            } else if (dstPrc == Precision::I64) {
                vcvtpd2qq(vSrc, vSrc);
            } else if (dstPrc == Precision::I32) {
                vcvtpd2dq(ymmSrc, vSrc);
            }
            break;
        case Precision::I64:
            if (dstPrc == Precision::FP32 || dstPrc == Precision::BF16) {
                vcvtqq2ps(ymmSrc, vSrc);
            } else if (dstPrc == Precision::FP64) {
                vcvtqq2pd(vSrc, vSrc);
            }
            break;
        case Precision::FP32:
            if (dstPrc == Precision::I64) {
                vcvtps2qq(vSrc, ymmSrc);
            } else if ((dstPrc == Precision::U8 || dstPrc == Precision::U16) && x64::mayiuse(x64::avx512_core)) {
                vcvtps2udq(vSrc, vSrc);
            } else if (dstPrc != Precision::FP32 && dstPrc != Precision::BF16) {
                uni_vcvtps2dq(vSrc, vSrc);
            }
            break;
        case Precision::I32:
            if (dstPrc == Precision::FP32 || dstPrc == Precision::BF16) {
                uni_vcvtdq2ps(vSrc, vSrc);
            }
            break;
        default:
            IE_THROW() << "Unsupported source precision: " << srcPrc;
    }

    switch (dstPrc) {
        case Precision::FP64:
            uni_vmovups(dstAdr, vSrc);
            break;
        case Precision::FP32:
            if (srcPrc.size() == 8) {
                uni_vmovups(dstAdr, ymmSrc);
            } else {
                uni_vmovups(dstAdr, vSrc);
            }
            break;
        case Precision::I64:
            uni_vmovups(dstAdr, vSrc);
            break;
        case Precision::I32:
            if (srcPrc == Precision::I64) {
                vpmovsqd(dstAdr, vSrc);
            } else if (srcPrc == Precision::FP64) {
                uni_vmovups(dstAdr, ymmSrc);
            } else {
                uni_vmovups(dstAdr, vSrc);
            }
            break;
        case Precision::BF16:
            vcvtneps2bf16->emit_code({static_cast<size_t>(ymmSrc.getIdx())}, {static_cast<size_t>(ymmSrc.getIdx())});
            vmovdqu16(dstAdr, ymmSrc);
            break;
        case Precision::I16:
            if (x64::mayiuse(x64::avx512_core)) {
                vpmovsdw(dstAdr, vSrc);
            } else {
                uni_vpackssdw(vSrc, vSrc, vSrc);
                if (x64::mayiuse(x64::avx)) {
                    vpermq(ymmSrc, ymmSrc, 0x08);
                    uni_vmovdqu(dstAdr, xmmSrc);
                } else {
                    movq(dstAdr, xmmSrc);
                }
            }
            break;
        case Precision::U16:
            if (x64::mayiuse(x64::avx512_core)) {
                vpmovusdw(dstAdr, xmmSrc);
            } else {
                uni_vpackusdw(vSrc, vSrc, vSrc);
                if (x64::mayiuse(x64::avx)) {
                    vpermq(ymmSrc, ymmSrc, 0x08);
                    uni_vmovdqu(dstAdr, xmmSrc);
                } else {
                    movq(dstAdr, xmmSrc);
                }
            }
            break;
        case Precision::I8:
            if (x64::mayiuse(x64::avx512_core)) {
                if (srcPrc == Precision::I64) {
                    vpmovsqb(dstAdr, vSrc);
                } else {
                    vpmovsdb(dstAdr, vSrc);
                }
            } else {
                uni_vpackssdw(vSrc, vSrc, vSrc);
                if (x64::mayiuse(x64::avx)) {
                    vpermq(ymmSrc, ymmSrc, 0x08);
                }
                uni_vpacksswb(vSrc, vSrc, vSrc);
                if (x64::mayiuse(x64::avx)) {
                    vmovq(dstAdr, xmmSrc);
                } else {
                    movd(dstAdr, xmmSrc);
                }
            }
            break;
        case Precision::U8:
            if (x64::mayiuse(x64::avx512_core)) {
                if (srcPrc == Precision::I64) {
                    vpmovusqb(dstAdr, vSrc);
                } else {
                    vpmovusdb(dstAdr, vSrc);
                }
            } else {
                uni_vpackusdw(vSrc, vSrc, vSrc);
                if (x64::mayiuse(x64::avx)) {
                    vpermq(ymmSrc, ymmSrc, 0x08);
                }
                uni_vpackuswb(vSrc, vSrc, vSrc);
                if (x64::mayiuse(x64::avx)) {
                    vmovq(dstAdr, xmmSrc);
                } else {
                    movd(dstAdr, xmmSrc);
                }
            }
            break;
        default:
            IE_THROW() << "Unsupported destination precision: " << dstPrc;
    }
}

void JitKernelBase::storeScalar(const Address &dstAdr,
                                const Xmm &vSrc,
                                const InferenceEngine::Precision &dstPrc,
                                const InferenceEngine::Precision &srcPrc) {
    switch (srcPrc) {
        case Precision::FP64:
            if (dstPrc == Precision::FP32) {
                uni_vcvtpd2ps(vSrc, vSrc);
            } else if (dstPrc == Precision::I64) {
                vcvtpd2qq(vSrc, vSrc);
            } else if (dstPrc == Precision::I32) {
                uni_vcvtpd2dq(vSrc, vSrc);
            }
            break;
        case Precision::I64:
            if (dstPrc == Precision::FP32 || dstPrc == Precision::BF16) {
                vcvtqq2ps(vSrc, vSrc);
            } else if (dstPrc == Precision::I32) {
                vpmovsqd(vSrc, vSrc);
            }
            break;
        case Precision::FP32:
            if (dstPrc == Precision::I64) {
                vcvtps2qq(vSrc, vSrc);
            } else if (dstPrc == Precision::U8 && x64::mayiuse(x64::avx512_core)) {
                vcvtps2udq(vSrc, vSrc);
            } else if (dstPrc != Precision::FP32 && dstPrc != Precision::BF16) {
                uni_vcvtps2dq(vSrc, vSrc);
            }
            break;
        case Precision::I32:
            if (dstPrc == Precision::FP32 || dstPrc == Precision::BF16) {
                uni_vcvtdq2ps(vSrc, vSrc);
            }
            break;
        default:
            IE_THROW() << "Unsupported source precision: " << srcPrc;
    }

    switch (dstPrc) {
        case Precision::FP64:
        case Precision::I64:
            uni_vmovsd(dstAdr, vSrc);
            break;
        case Precision::FP32:
        case Precision::I32:
            uni_vmovss(dstAdr, vSrc);
            break;
        case Precision::BF16:
            uni_vpsrld(vSrc, vSrc, 16);
            uni_vpextrw(dstAdr, vSrc, 0x0);
            break;
        case Precision::I16:
            uni_vpackssdw(vSrc, vSrc, vSrc);
            uni_vpextrw(dstAdr, vSrc, 0x0);
            break;
        case Precision::U16:
            uni_vpackusdw(vSrc, vSrc, vSrc);
            uni_vpextrw(dstAdr, vSrc, 0x0);
            break;
        case Precision::I8:
            if (x64::mayiuse(x64::avx512_core)) {
                vpmovsdb(vSrc, vSrc);
            } else {
                uni_vpackssdw(vSrc, vSrc, vSrc);
                uni_vpacksswb(vSrc, vSrc, vSrc);
            }
            uni_vpextrb(dstAdr, vSrc, 0x0);
            break;
        case Precision::U8:
            if (x64::mayiuse(x64::avx512_core)) {
                vpmovusdb(vSrc, vSrc);
            } else {
                uni_vpackusdw(vSrc, vSrc, vSrc);
                uni_vpackuswb(vSrc, vSrc, vSrc);
            }
            uni_vpextrb(dstAdr, vSrc, 0);
            break;
        default:
            IE_THROW() << "Unsupported destination precision: " << dstPrc;
    }
}
