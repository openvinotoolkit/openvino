// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel_base.hpp"

using namespace dnnl::impl::cpu;

namespace ov {
namespace intel_cpu {
namespace kernel {

JitKernelBase::JitKernelBase(const char* name, x64::cpu_isa_t isa)
        : x64::jit_generator(name, isa), m_isa(isa) {
    vlen = x64::isa_max_vlen(isa);
}

void JitKernelBase::uni_vfmsub132ps(const Xbyak::Xmm& v_dst,
                                    const Xbyak::Xmm& v_src,
                                    const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vfmsub132ps(v_dst, v_src, op);
    } else if (isValidIsa(x64::avx)) {
        assert(v_dst.getIdx() != v_src.getIdx());
        vmulps(v_dst, v_dst, op);
        vsubps(v_dst, v_dst, v_src);
    } else {
        assert(v_dst.getIdx() != v_src.getIdx());
        mulps(v_dst, op);
        subps(v_dst, v_src);
    }
}

void JitKernelBase::uni_vfnmadd132ps(const Xbyak::Xmm& v_dst,
                                     const Xbyak::Xmm& v_src,
                                     const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vfnmadd132ps(v_dst, v_src, op);
    } else if (isValidIsa(x64::avx)) {
        assert(v_dst.getIdx() != v_src.getIdx());
        vmulps(v_dst, v_dst, op);
        vsubps(v_dst, v_src, v_dst);
    } else {
        assert(v_dst.getIdx() != v_src.getIdx());
        mulps(v_dst, op);
        subps(v_src, v_dst);
        movups(v_dst, v_src);
    }
}

void JitKernelBase::uni_vfmsub231ps(const Xbyak::Xmm& v_dst,
                                    const Xbyak::Xmm& v_src,
                                    const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vfmsub231ps(v_dst, v_src, op);
    } else if (isValidIsa(x64::avx)) {
        assert(!v_dst.isEqualIfNotInherited(op));
        vmulps(v_src, v_src, op);
        vsubps(v_dst, v_src, v_dst);
    } else {
        assert(!v_dst.isEqualIfNotInherited(op));
        mulps(v_src, op);
        subps(v_src, v_dst);
        movups(v_dst, v_src);
    }
}

void JitKernelBase::uni_vpaddd(const Xbyak::Ymm& v_dst,
                               const Xbyak::Ymm& v_src,
                               const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vpaddd(v_dst, v_src, op);
    } else if (isValidIsa(x64::avx)) {
        Xbyak::Xmm xmmDst(v_dst.getIdx());
        vmovups(v_dst, v_src);
        if (op.isYMM()) {
            Xbyak::Ymm ymmOp(op.getIdx());
            Xbyak::Xmm xmmOp(op.getIdx());
            paddd(xmmDst, xmmOp);
            vperm2f128(v_dst, v_dst, v_dst, 0x1);
            vperm2f128(ymmOp, ymmOp, ymmOp, 0x1);
            paddd(xmmDst, xmmOp);
            vperm2f128(v_dst, v_dst, v_dst, 0x1);
            vperm2f128(ymmOp, ymmOp, ymmOp, 0x1);
        } else if (op.isMEM()) {
            const int vlen = x64::cpu_isa_traits<x64::sse41>::vlen;
            paddd(xmmDst, op.getAddress());
            vperm2f128(v_dst, v_dst, v_dst, 0x1);
            paddd(xmmDst, ptr[op.getAddress().getRegExp() + vlen]);
            vperm2f128(v_dst, v_dst, v_dst, 0x1);
        } else {
            OPENVINO_THROW("Not supported operand type.");
        }
    } else if (isValidIsa(x64::sse41)) {
        assert(v_dst.getIdx() != v_src.getIdx());
        paddd(v_dst, op);
    } else {
        OPENVINO_THROW("Not defined behavior for instruction 'vpaddd' in current instructions set.");
    }
}

void JitKernelBase::uni_vpaddq(const Xbyak::Xmm& v_dst,
                               const Xbyak::Xmm& v_src,
                               const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vpaddq(v_dst, v_src, op);
    } else {
        if (v_dst.getIdx() != v_src.getIdx()) {
            movups(v_dst, v_src);
        }
        paddq(v_dst, op);
    }
}

void JitKernelBase::uni_vpsubd(const Xbyak::Ymm& v_dst,
                               const Xbyak::Ymm& v_src,
                               const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vpsubd(v_dst, v_src, op);
    } else if (isValidIsa(x64::avx)) {
        Xbyak::Xmm xmmDst(v_dst.getIdx());
        vmovups(v_dst, v_src);
        if (op.isYMM()) {
            Xbyak::Ymm ymmOp(op.getIdx());
            Xbyak::Xmm xmmOp(op.getIdx());
            psubd(xmmDst, xmmOp);
            vperm2f128(v_dst, v_dst, v_dst, 0x1);
            vperm2f128(ymmOp, ymmOp, ymmOp, 0x1);
            psubd(xmmDst, xmmOp);
            vperm2f128(v_dst, v_dst, v_dst, 0x1);
            vperm2f128(ymmOp, ymmOp, ymmOp, 0x1);
        } else if (op.isMEM()) {
            const int vlen = x64::cpu_isa_traits<x64::sse41>::vlen;
            psubd(xmmDst, op.getAddress());
            vperm2f128(v_dst, v_dst, v_dst, 0x1);
            psubd(xmmDst, ptr[op.getAddress().getRegExp() + vlen]);
            vperm2f128(v_dst, v_dst, v_dst, 0x1);
        } else {
            OPENVINO_THROW("Not supported operand type.");
        }
    } else if (isValidIsa(x64::sse41)) {
        assert(v_dst.getIdx() != v_src.getIdx());
        psubd(v_dst, op);
    } else {
        OPENVINO_THROW("Not defined behavior for instruction 'vpsubd' in current instructions set.");
    }
}

void JitKernelBase::uni_vsubpd(const Xbyak::Xmm& v_dst,
                               const Xbyak::Xmm& v_src,
                               const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx)) {
        vsubpd(v_dst, v_src, op);
    } else {
        if (v_dst.getIdx() != v_src.getIdx()) {
            movups(v_dst, v_src);
        }
        subpd(v_dst, op);
    }
}

void JitKernelBase::uni_vmulpd(const Xbyak::Xmm& v_dst,
                               const Xbyak::Xmm& v_src,
                               const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx)) {
        vmulpd(v_dst, v_src, op);
    } else {
        if (v_dst.getIdx() != v_src.getIdx()) {
            movups(v_dst, v_src);
        }
        mulpd(v_dst, op);
    }
}

void JitKernelBase::uni_vpmuludq(const Xbyak::Xmm& v_dst,
                                 const Xbyak::Xmm& v_src,
                                 const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vpmuludq(v_dst, v_src, op);
    } else {
        if (v_dst.getIdx() != v_src.getIdx()) {
            movups(v_dst, v_src);
        }
        pmuludq(v_dst, op);
    }
}

void JitKernelBase::uni_vdivps(const Xbyak::Xmm& v_dst,
                               const Xbyak::Operand& op1,
                               const Xbyak::Operand& op2) {
    if (isValidIsa(x64::avx)) {
        vdivps(v_dst, op1, op2);
    } else {
        if (!v_dst.isEqualIfNotInherited(op1)) {
            movups(v_dst, op1);
        }
        divps(v_dst, op2);
    }
}

void JitKernelBase::uni_vdivpd(const Xbyak::Xmm& v_dst,
                               const Xbyak::Xmm& v_src,
                               const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx)) {
        vdivpd(v_dst, v_src, op);
    } else {
        if (v_dst.getIdx() != v_src.getIdx()) {
            movups(v_dst, v_src);
        }
        divpd(v_dst, op);
    }
}

void JitKernelBase::uni_vandps(const Xbyak::Xmm& v_dst,
                               const Xbyak::Xmm& vSrs,
                               const Xbyak::Operand &op) {
    if (isValidIsa(x64::avx)) {
        vandps(v_dst, vSrs, op);
    } else {
        if (!v_dst.isEqualIfNotInherited(vSrs)) {
            movups(v_dst, vSrs);
        }
        andps(v_dst, op);
    }
}

void JitKernelBase::uni_vandnps(const Xbyak::Xmm& v_dst,
                                const Xbyak::Xmm& vSrs,
                                const Xbyak::Operand &op) {
    if (isValidIsa(x64::avx)) {
        vandnps(v_dst, vSrs, op);
    } else {
        if (!v_dst.isEqualIfNotInherited(vSrs)) {
            movups(v_dst, vSrs);
        }
        andnps(v_dst, op);
    }
}

void JitKernelBase::gatherdd(const Xbyak::Xmm&    v_dst,
                             const Xbyak::Reg64&  rSrcPtr,
                             const Xbyak::Xmm&    vSrcShift,
                             const Xbyak::Opmask& kReadMask,
                             const bool useMask,
                             const bool zeroFill) {
    if (kReadMask.getIdx() == 0) {
        OPENVINO_THROW("The vpgatherdd instruction cannot use the register k0 as mask.");
    }
    if (!useMask)
        kxnord(kReadMask, kReadMask, kReadMask);
    if (zeroFill)
        uni_vpxor(v_dst, v_dst, v_dst);

    vpgatherdd(v_dst | kReadMask, ptr[rSrcPtr + vSrcShift]);
}

void JitKernelBase::gatherdd(const Xbyak::Xmm&   v_dst,
                             const Xbyak::Reg64& rSrcPtr,
                             const Xbyak::Xmm&   vSrcShift,
                             const Xbyak::Xmm&   vReadMask,
                             const bool useMask,
                             const bool zeroFill) {
    if (v_dst.getIdx() == vSrcShift.getIdx() || v_dst.getIdx() == vReadMask.getIdx() || vSrcShift.getIdx() == vReadMask.getIdx()) {
        OPENVINO_THROW("Any pair of the index, mask, or destination registers cannot be the same.");
    }
    if (zeroFill)
        pxor(v_dst, v_dst); // Don't use vpxor. It zeros the rest of the YMM register.

    if (isValidIsa(x64::avx2)) {
        if (!useMask)
            uni_vpcmpeqd(vReadMask, vReadMask, vReadMask);

        vpgatherdd(v_dst, ptr[rSrcPtr + vSrcShift], vReadMask);
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
            pinsrd(v_dst, ptr[rSrcPtr + rAux], i);

            if (useMask)
                L(lLoopNext);
        }
    }
}

void JitKernelBase::gatherdd(const Xbyak::Ymm&   v_dst,
                             const Xbyak::Reg64& rSrcPtr,
                             const Xbyak::Ymm&   vSrcShift,
                             const Xbyak::Ymm&   vReadMask,
                             const bool useMask,
                             const bool zeroFill) {
    if (v_dst.getIdx() == vSrcShift.getIdx() || v_dst.getIdx() == vReadMask.getIdx() || vSrcShift.getIdx() == vReadMask.getIdx()) {
        OPENVINO_THROW("Any pair of the index, mask, or destination registers cannot be the same.");
    }
    if (isValidIsa(x64::avx2)) {
        if (!useMask)
            uni_vpcmpeqd(vReadMask, vReadMask, vReadMask);
        if (zeroFill)
            uni_vpxor(v_dst, v_dst, v_dst);

        vpgatherdd(v_dst, ptr[rSrcPtr + vSrcShift], vReadMask);
    } else {
        Xbyak::Xmm xmmDst      = Xbyak::Xmm(v_dst.getIdx()),
                   xmmSrcShft  = Xbyak::Xmm(vSrcShift.getIdx()),
                   xmmReadMask = Xbyak::Xmm(vReadMask.getIdx());
        for (uint8_t i = 0; i < 2; i++) {
            gatherdd(xmmDst, rSrcPtr, xmmSrcShft, xmmReadMask, useMask, zeroFill);

            vperm2f128(v_dst, v_dst, v_dst, 0x1);
            vperm2f128(vSrcShift, vSrcShift, vSrcShift, 0x1);
            if (useMask)
                vperm2f128(vReadMask, vReadMask, vReadMask, 0x1);
        }
    }
}

void JitKernelBase::uni_vpbroadcastq(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
    if (isValidIsa(x64::avx2)) {
        vpbroadcastq(x, op);
    } else {
        movsd(x, op);
        shufpd(x, x, 0x0);
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

void JitKernelBase::uni_vroundpd(const Xbyak::Xmm& v_dst, const Xbyak::Operand& op, const uint8_t imm) {
    if (isValidIsa(x64::avx512_core)) {
        vrndscalepd(v_dst, op, imm & 0x3);
    } else if (isValidIsa(x64::avx)) {
        vroundpd(v_dst, op, imm);
    } else {
        roundpd(v_dst, op, imm);
    }
}

void JitKernelBase::uni_vcvtdq2pd(const Xbyak::Xmm& v_dst,
                                  const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx)) {
        vcvtdq2pd(v_dst, op);
    } else {
        cvtdq2pd(v_dst, op);
    }
}

void JitKernelBase::uni_vcvtpd2dq(const Xbyak::Xmm& v_dst,
                                  const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx)) {
        vcvtpd2dq(v_dst, op);
    } else {
        cvtpd2dq(v_dst, op);
    }
}

void JitKernelBase::uni_vpmovzxdq(const Xbyak::Xmm& v_dst,
                                  const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vpmovzxdq(v_dst, op);
    } else {
        pmovzxdq(v_dst, op);
    }
}

void JitKernelBase::uni_vshufpd(const Xbyak::Xmm& v_dst,
                                const Xbyak::Xmm& v_src,
                                const Xbyak::Operand& op,
                                uint8_t imm) {
    if (isValidIsa(x64::avx)) {
        vshufpd(v_dst, v_src, op, imm);
    } else {
        if (v_dst.getIdx() != v_src.getIdx()) {
            movups(v_dst, v_src);
        }
        shufpd(v_dst, op, imm);
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
    if (!one_of(typeSize, 1u, 2u, 4u, 8u)) {
        OPENVINO_THROW("Could not fill data with type size ", typeSize);
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
    if (!one_of(typeSize, 1u, 2u, 4u, 8u)) {
        OPENVINO_THROW("Could not fill data with type size ", typeSize);
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

void JitKernelBase::load(const Xbyak::Xmm&     v_dst,
                         const Xbyak::Address& srcAddr,
                         const Xbyak::Reg64&   rLoadNum,
                         const size_t          typeSize,
                         const bool            zeroFilling) {
    if (!one_of(typeSize, 1u, 2u, 4u, 8u)) {
        OPENVINO_THROW("Could not load data with type size ", typeSize);
    }
    const uint8_t elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;
    Xbyak::Label lEnd;
    if (zeroFilling)
        pxor(v_dst, v_dst);

    for (uint8_t i = 0; i < elPerVec; i++) {
        cmp(rLoadNum, i);
        jle(lEnd, T_NEAR);

        const size_t offset = i * typeSize;
        if (typeSize == 1)
            pinsrb(v_dst, ptr[srcAddr.getRegExp() + offset], i);
        else if (typeSize == 2)
            pinsrw(v_dst, ptr[srcAddr.getRegExp() + offset], i);
        else if (typeSize == 4)
            pinsrd(v_dst, ptr[srcAddr.getRegExp() + offset], i);
        else if (typeSize == 8)
            pinsrq(v_dst, ptr[srcAddr.getRegExp() + offset], i);
    }
    L(lEnd);
}

void JitKernelBase::load(const Xbyak::Ymm&     v_dst,
                         const Xbyak::Address& srcAddr,
                         const Xbyak::Reg64&   rLoadNum,
                         const size_t          typeSize,
                         const bool            zeroFilling) {
    if (!one_of(typeSize, 1u, 2u, 4u, 8u)) {
        OPENVINO_THROW("Could not load data with type size ", typeSize);
    }
    const size_t elPerXmm = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;
    Xbyak::Label lEnd;
    if (zeroFilling)
        uni_vpxor(v_dst, v_dst, v_dst);
    Xbyak::Xmm xmmDst(v_dst.getIdx());

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
        vperm2f128(v_dst, v_dst, v_dst, 0x1);
    }
    L(lEnd);
}

void JitKernelBase::store(const Xbyak::Address& dstAddr,
                          const Xbyak::Xmm&     v_src,
                          const Xbyak::Reg64&   rToStoreNum,
                          const size_t          typeSize) {
    if (!one_of(typeSize, 1u, 2u, 4u, 8u)) {
        OPENVINO_THROW("Could not store data with type size ", typeSize);
    }
    Xbyak::Label lEnd;
    const size_t elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;

    for (size_t i = 0; i < elPerVec; i++) {
        cmp(rToStoreNum, i);
        jle(lEnd, T_NEAR);

        const size_t offset = i * typeSize;
        if (typeSize == 1) {
            uni_vpextrb(ptr[dstAddr.getRegExp() + offset], v_src, i);
        } else if (typeSize == 2) {
            uni_vpextrw(ptr[dstAddr.getRegExp() + offset], v_src, i);
        } else if (typeSize == 4) {
            uni_vpextrd(ptr[dstAddr.getRegExp() + offset], v_src, i);
        } else if (typeSize == 8) {
            uni_vpextrq(ptr[dstAddr.getRegExp() + offset], v_src, i);
        }
    }
    L(lEnd);
}

void JitKernelBase::store(const Xbyak::Address& dstAddr,
                          const Xbyak::Ymm&     v_src,
                          const Xbyak::Reg64&   rToStoreNum,
                          const size_t          typeSize) {
    if (!one_of(typeSize, 1u, 2u, 4u, 8u)) {
        OPENVINO_THROW("Could not store data with type size ", typeSize);
    }
    Xbyak::Label lEnd;
    Xbyak::Xmm xmmSrc(v_src.getIdx());
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
        vperm2f128(v_src, v_src, v_src, 0x1);
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

} // namespace kernel
} // namespace intel_cpu
} // namespace ov
