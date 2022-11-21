// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_base.hpp"
#include "ie/ie_common.h"
#include "utils/general_utils.h"

using namespace ov;
using namespace intel_cpu;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;


void jit_base::push(const Xbyak::Xmm& xmm) {
    if (xmm.isXMM()) {
        sub(rsp, xmm_len);
        uni_vmovdqu(ptr[rsp], xmm);
    }
    else if (xmm.isYMM()) {
        sub(rsp, ymm_len);
        uni_vmovdqu(ptr[rsp], Xbyak::Ymm{ xmm.getIdx() });
    }
    else if (xmm.isZMM()) {
        sub(rsp, zmm_len);
        uni_vmovdqu(ptr[rsp], Xbyak::Zmm{ xmm.getIdx() });
    }
}

void jit_base::pop(const Xbyak::Xmm& xmm) {
    if (xmm.isXMM()) {
        uni_vmovdqu(xmm, ptr[rsp]);
        add(rsp, xmm_len);
    }
    else if (xmm.isYMM()) {
        uni_vmovdqu(Xbyak::Ymm{ xmm.getIdx() }, ptr[rsp]);
        add(rsp, ymm_len);
    }
    else if (xmm.isZMM()) {
        uni_vmovdqu(Xbyak::Zmm{ xmm.getIdx() }, ptr[rsp]);
        add(rsp, zmm_len);
    }
}

void jit_base::uni_vaddps(const Xbyak::Xmm& x,
    const Xbyak::Xmm& op1,
    const Xbyak::Operand& op2) {
    if (is_valid_isa(x64::avx)) {
        vaddps(x, op1, op2);
    }
    else {
        if (x.getIdx() == op1.getIdx()) {
            addps(x, op2);
        }
        else if (x.isEqualIfNotInherited(op2)) {
            addps(x, op1);
        }
        else {
            movups(x, op1);
            addps(x, op2);
        }
    }
}

void jit_base::uni_vsubps(const Xbyak::Xmm& x,
    const Xbyak::Xmm& op1,
    const Xbyak::Operand& op2) {
    if (is_valid_isa(x64::avx)) {
        vsubps(x, op1, op2);
    }
    else {
        if (x.getIdx() == op1.getIdx()) {
            subps(x, op2);
        }
        else if (x.isEqualIfNotInherited(op2)) {
            push(op1);
            subps(op1, op2);
            movups(x, op1);
            pop(op1);
        }
        else {
            movups(x, op1);
            subps(x, op2);
        }
    }
}

void jit_base::uni_vcmpps(const Xbyak::Xmm& x,
    const Xbyak::Xmm& op1,
    const Xbyak::Operand& op2,
    const int cmp_predicate) {
    if (is_valid_isa(x64::avx)) {
        vcmpps(x, op1, op2, cmp_predicate);
    }
    else {
        if (x.getIdx() == op1.getIdx()) {
            cmpps(x, op2, cmp_predicate);
        }
        else if (x.isEqualIfNotInherited(op2)) {
            push(op1);
            cmpps(op1, op2, cmp_predicate);
            movups(x, op1);
            pop(op1);
        }
        else {
            movups(x, op1);
            cmpps(x, op2, cmp_predicate);
        }
    }
}

void jit_base::uni_vfmsub132ps(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrc, const Xbyak::Operand& op) {
    if (is_valid_isa(x64::avx2)) {
        vfmsub132ps(vDst, vSrc, op);
    } else if (is_valid_isa(x64::avx)) {
        assert(vDst.getIdx() != vSrc.getIdx());
        vmulps(vDst, vDst, op);
        vsubps(vDst, vDst, vSrc);
    } else {
        assert(vDst.getIdx() != vSrc.getIdx());
        mulps(vDst, op);
        subps(vDst, vSrc);
    }
}

void jit_base::uni_vfnmadd132ps(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrc, const Xbyak::Operand& op) {
    if (is_valid_isa(x64::avx2)) {
        vfnmadd132ps(vDst, vSrc, op);
    } else if (is_valid_isa(x64::avx)) {
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

void jit_base::uni_vfmsub231ps(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrc, const Xbyak::Operand& op) {
    if (is_valid_isa(x64::avx2)) {
        vfmsub231ps(vDst, vSrc, op);
    } else if (is_valid_isa(x64::avx)) {
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

void jit_base::uni_vpaddd(const Xbyak::Ymm& vDst,
    const Xbyak::Ymm& vSrc,
    const Xbyak::Operand& op) {
    if (is_valid_isa(x64::avx2)) {
        vpaddd(vDst, vSrc, op);
    }
    else if (is_valid_isa(x64::avx)) {
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
        }
        else if (op.isMEM()) {
            const int vlen = x64::cpu_isa_traits<x64::sse41>::vlen;
            paddd(xmmDst, op.getAddress());
            vperm2f128(vDst, vDst, vDst, 0x1);
            paddd(xmmDst, ptr[op.getAddress().getRegExp() + vlen]);
            vperm2f128(vDst, vDst, vDst, 0x1);
        }
        else {
            IE_THROW() << "Not supported operand type.";
        }
    }
    else if (is_valid_isa(x64::sse41)) {
        assert(vDst.getIdx() != vSrc.getIdx());
        paddd(vDst, op);
    }
    else {
        IE_THROW() << "Not defined behavior for instruction 'vpaddd' in current instructions set.";
    }
}

void jit_base::uni_vpsubd(const Xbyak::Ymm& vDst,
    const Xbyak::Ymm& vSrc,
    const Xbyak::Operand& op) {
    if (is_valid_isa(x64::avx2)) {
        vpsubd(vDst, vSrc, op);
    }
    else if (is_valid_isa(x64::avx)) {
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
        }
        else if (op.isMEM()) {
            const int vlen = x64::cpu_isa_traits<x64::sse41>::vlen;
            psubd(xmmDst, op.getAddress());
            vperm2f128(vDst, vDst, vDst, 0x1);
            psubd(xmmDst, ptr[op.getAddress().getRegExp() + vlen]);
            vperm2f128(vDst, vDst, vDst, 0x1);
        }
        else {
            IE_THROW() << "Not supported operand type.";
        }
    }
    else if (is_valid_isa(x64::sse41)) {
        assert(vDst.getIdx() != vSrc.getIdx());
        psubd(vDst, op);
    }
    else {
        IE_THROW() << "Not defined behavior for instruction 'vpsubd' in current instructions set.";
    }
}

void jit_base::uni_vdivps(const Xbyak::Xmm& vDst,
    const Xbyak::Operand& op1,
    const Xbyak::Operand& op2) {
    if (is_valid_isa(x64::avx)) {
        vdivps(vDst, op1, op2);
    }
    else {
        if (!vDst.isEqualIfNotInherited(op1)) {
            movups(vDst, op1);
        }
        divps(vDst, op2);
    }
}

void jit_base::uni_vandps(const Xbyak::Xmm& vDst,
    const Xbyak::Xmm& vSrs,
    const Xbyak::Operand& op) {
    if (is_valid_isa(x64::avx)) {
        vandps(vDst, vSrs, op);
    }
    else {
        if (!vDst.isEqualIfNotInherited(vSrs)) {
            movups(vDst, vSrs);
        }
        andps(vDst, op);
    }
}

void jit_base::uni_vandnps(const Xbyak::Xmm& vDst,
    const Xbyak::Xmm& vSrs,
    const Xbyak::Operand& op) {
    if (is_valid_isa(x64::avx)) {
        vandnps(vDst, vSrs, op);
    }
    else {
        if (!vDst.isEqualIfNotInherited(vSrs)) {
            movups(vDst, vSrs);
        }
        andnps(vDst, op);
    }
}

void jit_base::uni_vpbroadcastd(const Xbyak::Xmm& x, const Xbyak::Operand& op) {
    if (is_valid_isa(x64::avx2)) {
        vpbroadcastd(x, op);
    }
    else if (is_valid_isa(x64::avx)) {
        if (op.isMEM()) {
            vbroadcastss(x, op.getAddress());
        }
        else {
            vmovss(x, x, op);
            vpshufd(x, x, 0x0);
        }
    }
    else {
        movss(x, op);
        pshufd(x, x, 0x0);
    }
}

void jit_base::uni_vpbroadcastd(const Xbyak::Ymm& x, const Xbyak::Operand& op) {
    if (is_valid_isa(x64::avx2)) {
        vpbroadcastd(x, op);
    }
    else {
        if (op.isMEM()) {
            vbroadcastss(x, op.getAddress());
        }
        else {
            const Xbyak::Xmm t(x.getIdx());
            if (!t.isEqualIfNotInherited(op)) {
                vmovss(t, t, op);
            }
            vinsertf128(x, x, t, 1);
            vshufps(x, x, x, 0);
        }
    }
}

void jit_base::load(const Xbyak::Xmm& vDst,
    const Xbyak::Address& srcAddr,
    const Xbyak::Reg64& rLoadNum,
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

void jit_base::load(const Xbyak::Ymm& vDst,
    const Xbyak::Address& srcAddr,
    const Xbyak::Reg64& rLoadNum,
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

void jit_base::store(const Xbyak::Address& dstAddr,
    const Xbyak::Xmm& vSrc,
    const Xbyak::Reg64& rToStoreNum,
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
        }
        else if (typeSize == 2) {
            uni_vpextrw(ptr[dstAddr.getRegExp() + offset], vSrc, i);
        }
        else if (typeSize == 4) {
            uni_vpextrd(ptr[dstAddr.getRegExp() + offset], vSrc, i);
        }
        else if (typeSize == 8) {
            uni_vpextrq(ptr[dstAddr.getRegExp() + offset], vSrc, i);
        }
    }
    L(lEnd);
}

void jit_base::store(const Xbyak::Address& dstAddr,
    const Xbyak::Ymm& vSrc,
    const Xbyak::Reg64& rToStoreNum,
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
            }
            else if (typeSize == 4) {
                uni_vpextrd(ptr[dstAddr.getRegExp() + offset], xmmSrc, j);
            }
            else if (typeSize == 2) {
                uni_vpextrw(ptr[dstAddr.getRegExp() + offset], xmmSrc, j);
            }
            else if (typeSize == 1) {
                uni_vpextrb(ptr[dstAddr.getRegExp() + offset], xmmSrc, j);
            }
        }

        L(lPerm);
        vperm2f128(vSrc, vSrc, vSrc, 0x1);
    }
    L(lEnd);
}
