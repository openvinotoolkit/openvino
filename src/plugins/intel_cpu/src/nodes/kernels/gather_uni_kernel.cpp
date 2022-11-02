// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_uni_kernel.hpp"
#include <ie_common.h>

using namespace dnnl::impl::cpu;

namespace ov {
namespace intel_cpu {

#define GET_OFF(field) offsetof(gatherJitExecArgs, field)


template<x64::cpu_isa_t isa>
void jitGatherKernelBase<isa>::initialize(const jGatherConfParams& jcp) {
    this->beforeAxisSize = jcp.beforeAxisSize;
    this->specIdxSize = jcp.specIdxSize;
    this->batchDims = jcp.batchDims;
    this->reverseIndexing = jcp.reverseIndexing;
    this->afterAxisSize = jcp.afterAxisSize;
    this->dynamicShapes = jcp.dynamicShapes;
}

template<x64::cpu_isa_t isa>
bool jitGatherKernelBase<isa>::isSameParams(const jGatherConfParams& jcp) {
    return beforeAxisSize == jcp.beforeAxisSize && specIdxSize == jcp.specIdxSize && batchDims == jcp.batchDims &&
           afterAxisSize == jcp.afterAxisSize;
}

template<x64::cpu_isa_t isa>
void jitGatherKernelBase<isa>::create_ker() {
    auto code = x64::jit_generator::create_kernel();
    if (code != dnnl::impl::status::success)
        IE_THROW() << "Could not create Gather kernel. Error code: " << std::to_string(code);
    ker_ = (decltype(ker_))jit_ker();
}

template <x64::cpu_isa_t isa>
void jitGatherKernelBase<isa>::generate() {
    this->preamble();
    mov(regSrc, ptr[regParams + GET_OFF(src)]);
    mov(regDst, ptr[regParams + GET_OFF(dst)]);
    mov(regIndices, ptr[regParams + GET_OFF(indices)]);
    mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);
    uni_vpxor(vmmZeros, vmmZeros, vmmZeros);
    uploadParamPtrWithVpbroadcastd(vmmAxisDim, GET_OFF(axisDim));
    if (dynamicShapes) {
        generateForDynamicShapes();
    } else {
        generateForStaticShapes();
    }
    this->postamble();
}

template <x64::cpu_isa_t isa>
void jitGatherKernelBase<isa>::uploadParamPtrWithVpbroadcastd(const Vmm& vmmDest, size_t offset) {
    RegistersPool::Reg<Xbyak::Reg64> regAux {regPool};
    mov(regAux, ptr[regParams + offset]);
    uni_vpbroadcastd(vmmDest, ptr[regAux]);
}

template <x64::cpu_isa_t isa>
void jitGatherKernelBase<isa>::uploadParamPtrWithVmovups(const Vmm& vmmDest, size_t offset) {
    RegistersPool::Reg<Xbyak::Reg64> regAux {regPool};
    mov(regAux, ptr[regParams + offset]);
    uni_vmovups(vmmDest, ptr[regAux]);
}


template <x64::cpu_isa_t isa>
void jitGatherKernelBase<isa>::process(ShiftCalculator& shiftCalculator) {
    Xbyak::Label lTailProc, lEndProc;
    cmp(regWorkAmount, getDataElPerVec());
    jl(lTailProc, T_NEAR);
    processDataTypeSpecific(shiftCalculator);
    jmp(lEndProc, T_NEAR);
    L(lTailProc);
    tail(shiftCalculator, false);
    L(lEndProc);
}

template <>
void jitGatherKernelBase<x64::avx2>::combineMasks(Vmask& maskA, const Vmask& maskB) {
    vpand(maskA, maskA, maskB);
}
template <>
void jitGatherKernelBase<x64::avx512_core>::combineMasks(Vmask& maskA, const Vmask& maskB) {
    kandd(maskA, maskA, maskB);
}

template <x64::cpu_isa_t isa>
void jitGatherKernelBase<isa>::tail(ShiftCalculator& shiftCalculator, bool shiftFirst) {
    Xbyak::Label lEnd;

    const int secondStepCycles = 4 / getDataTypeSize();
    for (int p = 0; p < secondStepCycles; p++) {
        cmp(regWorkAmount, 0);
        jle(lEnd, T_NEAR);

        RegistersPool::Reg<Vmm> vSrcShift;
        RegistersPool::Reg<Vmask> kGatherMask;
        std::tie(kGatherMask, vSrcShift) = shiftCalculator.calcSrcShift(*this, p > 0 || shiftFirst);

        RegistersPool::Reg<Vmask> kAuxMask1 {regPool};
        fillRestWorkMask(kAuxMask1, regWorkAmount);
        combineMasks(kGatherMask, kAuxMask1);
        RegistersPool::Reg<Vmm> vmmSrc {regPool};
        uni_vmovups(vmmSrc, vmmZeros);
        uniVpGatherDd(vmmSrc, ptr[regSrc + vSrcShift], kGatherMask);
        if (getDataTypeSize() == 4) {
            uni_vmovups_tail(ptr[regDst], kAuxMask1, vmmSrc);
            sub(regWorkAmount, getDataElPerVec());
        } else {
            vSrcShift.release();
            storeVectorPart(regDst, regWorkAmount, vmmSrc);
        }
    }
    L(lEnd);
}

template <x64::cpu_isa_t isa>
poolVmm<isa> jitGatherKernelBase<isa>::shiftIdxAndGather(ShiftCalculator& shiftCalculator, bool shiftFirst) {
    RegistersPool::Reg<Vmm> vmmCalculatedShifts;
    RegistersPool::Reg<Vmask> kGatherMask;
    std::tie(kGatherMask, vmmCalculatedShifts) = shiftCalculator.calcSrcShift(*this, shiftFirst);
    RegistersPool::Reg<Vmm> vmmGatheredData {regPool};
    uni_vmovups(vmmGatheredData, vmmZeros);
    uniVpGatherDd(vmmGatheredData, ptr[regSrc + vmmCalculatedShifts], kGatherMask);
    return vmmGatheredData;
}

template <>
void jitGatherKernelBase<x64::avx2>::uniVpGatherDd(Vmm& vDst, const Xbyak::Address& srcAddr, Vmask& kMask) {
    vpgatherdd(vDst, srcAddr, kMask);
}
template <>
void jitGatherKernelBase<x64::avx512_core>::uniVpGatherDd(Vmm& vDst, const Xbyak::Address& srcAddr, Vmask& kMask) {
    vpgatherdd(vDst | kMask, srcAddr);
}

template <>
poolVmask<x64::avx2> jitGatherKernelBase<x64::avx2>::normalizeIndexesAndCalcShifts(Vmm& vRawIndices, poolVmask<x64::avx2> kDstMask) {
    // Compensate negative indices.
    if (reverseIndexing) {
        RegistersPool::Reg<Vmask> zerosOrAxisDimForNegativeIndexes {regPool};
        vpcmpgtd(zerosOrAxisDimForNegativeIndexes, vmmZeros, vRawIndices);
        vpand(zerosOrAxisDimForNegativeIndexes, zerosOrAxisDimForNegativeIndexes, vmmAxisDim);

        uni_vpaddd(vRawIndices, vRawIndices, zerosOrAxisDimForNegativeIndexes);
    }
    // Check boundaries.
    if (!kDstMask.isInitialized()) {
        kDstMask = RegistersPool::Reg<Vmask>{regPool};
    }
    vpcmpgtd(kDstMask, vmmAxisDim, vRawIndices);
    RegistersPool::Reg<Vmask> negativeIndexesMask {regPool};
    vpcmpgtd(negativeIndexesMask, vmmZeros, vRawIndices);
    vpandn(kDstMask, negativeIndexesMask, kDstMask);
    // Multiply by type size.
    if (getDataTypeSize() > 1)
        uni_vpslld(vRawIndices, vRawIndices, getDataTypeShift());
    return kDstMask;
}

template <>
poolVmask<x64::avx512_core> jitGatherKernelBase<x64::avx512_core>::normalizeIndexesAndCalcShifts(
        Vmm& vRawIndices, poolVmask<x64::avx512_core> kDstMask) {
    RegistersPool::Reg<Vmask> kAuxMask {regPool};
    // Compensate negative indices.
    if (reverseIndexing) {
        vpcmpgtd(kAuxMask, vmmZeros, vRawIndices);
        uni_vpaddd(vRawIndices | kAuxMask, vRawIndices, vmmAxisDim);
    }
    // Check boundaries.
    if (!kDstMask.isInitialized()) {
        kDstMask = RegistersPool::Reg<Vmask>{regPool};
    }
    vpcmpgtd(kAuxMask, vmmAxisDim, vRawIndices);
    vpcmpd(static_cast<Vmask&>(kDstMask) | kAuxMask, vmmZeros, vRawIndices, 2); // 2 - LE
    // Multiply by type size.
    if (getDataTypeSize() > 1) {
        uni_vpslld(vRawIndices, vRawIndices, getDataTypeShift());
    }
    return kDstMask;
}


template <>
void jitGatherKernelBase<x64::avx512_core>::fillRestWorkMask(Vmask& kDstMask, const Xbyak::Reg& rWorkRest) {
    Xbyak::Label lKmov;
    RegistersPool::Reg<Xbyak::Reg32> rOnes {regPool};
    mov(rOnes, 0x0000FFFF);
    cmp(rWorkRest, getIdxElPerVec());
    jge(lKmov);
    Xbyak::Reg8 rShift(Xbyak::Operand::CL);
    mov(rShift, getIdxElPerVec());
    sub(rShift, rWorkRest);
    shr(rOnes, rShift);
    L(lKmov);
    kmovw(kDstMask, rOnes);
}

template <>
void jitGatherKernelBase<x64::avx2>::fillRestWorkMask(Vmask& kDstMask, const Xbyak::Reg& rWorkRest) {
    Xbyak::Label lEnd;
    RegistersPool::Reg<Xbyak::Reg64> rAux0 {regPool};
    mov(rAux0, rWorkRest);
    RegistersPool::Reg<Xbyak::Reg32> rOnes {regPool};
    mov(rOnes, 0xFFFFFFFF);
    RegistersPool::Reg<Xbyak::Xmm> xmmAux{regPool};
    uni_vmovups(kDstMask, vmmZeros);
    for (uint8_t i = 0; i < getIdxElPerVec(); i++) {
        cmp(rAux0, 0);
        je(lEnd, T_NEAR);

        if (i % 4 == 0)
            uni_vmovups(xmmAux, Xbyak::Xmm(vmmZeros.getIdx()));

        vpinsrd(xmmAux, xmmAux, rOnes, i % 4);
        vinserti128(kDstMask, kDstMask, xmmAux, i / 4);
        sub(rAux0, 1);
    }
    L(lEnd);
}

template <x64::cpu_isa_t isa>
void jitGatherKernelBase<isa>::storeVectorPart(const Xbyak::Reg& rDst, const Xbyak::Reg& rToStoreCounter, Vmm& vmmSrc) {
    static const uint32_t vlenXmm = x64::cpu_isa_traits<x64::sse41>::vlen;
    Xbyak::Label lEnd;
    RegistersPool::Reg<Xbyak::Xmm> xAux {regPool};
    for (int j = 0; j < getVecLen() / vlenXmm; j++) {
        if (isa == x64::avx2)
            vextracti128(xAux, vmmSrc, j);
        else if (isa == x64::avx512_core)
            vextracti64x2(xAux, vmmSrc, j);

        for (int k = 0; k < 4; k++) {
            cmp(rToStoreCounter, 0);
            jle(lEnd, T_NEAR);

            if (getDataTypeSize() == 4)
                uni_vpextrd(ptr[rDst], xAux, k);
            else if (getDataTypeSize() == 2)
                uni_vpextrw(ptr[rDst], xAux, k * 2);
            else if (getDataTypeSize() == 1)
                uni_vpextrb(ptr[rDst], xAux, k * 4);

            add(rDst, getDataTypeSize());
            sub(rToStoreCounter, 1);
        }
    }
    L(lEnd);
}

template<x64::cpu_isa_t isa>
void jitGatherKernelForDataTypeSize<isa, DataType32bit>::processDataTypeSpecific(
        typename jitGatherKernelBase<isa>::ShiftCalculator& shiftCalculator) {
    Xbyak::Label lDstIdxLoop, lTail;

    // First iteration
    this->uni_vmovups(this->ptr[this->regDst], this->shiftIdxAndGather(shiftCalculator, false));

    // Main loop
    this->L(lDstIdxLoop);
    {
        this->add(this->regDst, this->getVecLen());
        this->sub(this->regWorkAmount, this->getDataElPerVec());
        this->cmp(this->regWorkAmount, this->getDataElPerVec());
        this->jl(lTail, this->T_NEAR);

        this->uni_vmovups(this->ptr[this->regDst], this->shiftIdxAndGather(shiftCalculator, true));

        this->jmp(lDstIdxLoop, this->T_NEAR);
    }

    this->L(lTail);
    this->tail(shiftCalculator, true);
}


template<x64::cpu_isa_t isa>
void jitGatherKernelForDataTypeSize<isa, DataType16bit>::processDataTypeSpecific(
        typename jitGatherKernelBase<isa>::ShiftCalculator& shiftCalculator) {
    static const unsigned shufMask16bitUni[16] = {0x05040100, 0x0D0C0908, 0x80808080, 0x80808080, 0x05040100, 0x0D0C0908, 0x80808080, 0x80808080,
                                                  0x05040100, 0x0D0C0908, 0x80808080, 0x80808080, 0x05040100, 0x0D0C0908, 0x80808080, 0x80808080};
    static const unsigned permMask16bitA2[8]   = {0, 1, 4, 5, 2, 3, 6, 7};
    static const unsigned permMask16bitA5[16]  = {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15};

    Xbyak::Label lDstIdxLoop1, lTail;
    RegistersPool::Reg<Vmm> vPermMask {this->regPool};
    RegistersPool::Reg<Vmm> vShufMask {this->regPool};
    RegistersPool::Reg<Vmm> vBuff0 {this->regPool};

    RegistersPool::Reg<Xbyak::Reg64> regAux1 {this->regPool};
    this->mov(regAux1, reinterpret_cast<uintptr_t>(shufMask16bitUni));
    this->uni_vmovups(vShufMask, this->ptr[regAux1]);
    if (isa == x64::avx2) {
        this->mov(regAux1, reinterpret_cast<uintptr_t>(permMask16bitA2));
    } else if (isa == x64::avx512_core) {
        this->mov(regAux1, reinterpret_cast<uintptr_t>(permMask16bitA5));
    }
    this->uni_vmovups(vPermMask, this->ptr[regAux1]);

    // First iteration
    this->vpshufb(vBuff0, this->shiftIdxAndGather(shiftCalculator, false), vShufMask);

    RegistersPool::Reg<Vmm> vAux0 {this->regPool};
    this->vpshufb(vAux0, this->shiftIdxAndGather(shiftCalculator, true), vShufMask);

    this->vshufps(vAux0, vBuff0, vAux0, 0x44);
    this->vpermd(vAux0, vPermMask, vAux0);

    this->uni_vmovups(this->ptr[this->regDst], vAux0);

    // Main loop.
    this->L(lDstIdxLoop1);
    {
        this->add(this->regDst, this->getVecLen());
        this->sub(this->regWorkAmount, this->getDataElPerVec());
        this->cmp(this->regWorkAmount, this->getDataElPerVec());
        this->jl(lTail, this->T_NEAR);

        this->vpshufb(vBuff0, this->shiftIdxAndGather(shiftCalculator, true), vShufMask);

        this->vpshufb(vAux0, this->shiftIdxAndGather(shiftCalculator, true), vShufMask);

        this->vshufps(vAux0, vBuff0, vAux0, 0x44);
        this->vpermd(vAux0, vPermMask, vAux0);

        this->uni_vmovups(this->ptr[this->regDst], vAux0);

        this->jmp(lDstIdxLoop1, this->T_NEAR);
    }

    this->L(lTail);
    this->tail(shiftCalculator, true);
}

template<x64::cpu_isa_t isa>
poolVmm<isa> jitGatherKernelForDataTypeSize<isa, DataType8bit>::calculateIdxShiftsForHalfIteration(
        typename jitGatherKernelBase<isa>::ShiftCalculator& shiftCalculator, Vmm& vShufMask, bool shiftFirst, poolVmm<isa> halfPart) {
    if (!halfPart.isInitialized()) {
        halfPart = RegistersPool::Reg<Vmm>{this->regPool};
    }
    this->vpshufb(halfPart, this->shiftIdxAndGather(shiftCalculator, shiftFirst), vShufMask);
    {
        auto gatheredData = this->shiftIdxAndGather(shiftCalculator, true);
        RegistersPool::Reg<Vmm> vAux0{this->regPool};
        this->vpshufb(vAux0, gatheredData, vShufMask);
        this->vshufps(halfPart, halfPart, vAux0, 0x0);
    }
    return halfPart;
}

template<x64::cpu_isa_t isa>
void jitGatherKernelForDataTypeSize<isa, DataType8bit>::processDataTypeSpecific(
        typename jitGatherKernelBase<isa>::ShiftCalculator& shiftCalculator) {
    static const unsigned shufMask8bitUni[16]  = {0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080,
                                                  0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080};
    static const unsigned permMask8bitA2[8]    = {0, 4, 1, 5, 2, 6, 3, 7};
    static const unsigned permMask8bitA5[16]   = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
    Xbyak::Label lDstIdxLoop1, lTail;

    {
        RegistersPool::Reg<Vmm> vShufMask {this->regPool};

        { // scope for regAux1
            RegistersPool::Reg<Xbyak::Reg64> regAux1{this->regPool};
            this->mov(regAux1, reinterpret_cast<uintptr_t>(shufMask8bitUni));
            this->uni_vmovups(vShufMask, this->ptr[regAux1]);
        }

        // First iteration
        RegistersPool::Reg<Vmm> vBuff0 = calculateIdxShiftsForHalfIteration(shiftCalculator, vShufMask, false);
        RegistersPool::Reg<Vmm> vBuff1 = calculateIdxShiftsForHalfIteration(shiftCalculator, vShufMask, true);
        RegistersPool::Reg<Vmm> vPermMask{this->regPool};
        {
            RegistersPool::Reg<Vmm> vAux0{this->regPool};
            this->vshufps(vAux0, vBuff0, vBuff1, 0x88);

            { // scope for regAux1
                RegistersPool::Reg<Xbyak::Reg64> regAux1{this->regPool};
                if (isa == x64::avx2) {
                    this->mov(regAux1, reinterpret_cast<uintptr_t>(permMask8bitA2));
                } else if (isa == x64::avx512_core) {
                    this->mov(regAux1, reinterpret_cast<uintptr_t>(permMask8bitA5));
                }
                this->uni_vmovups(vPermMask, this->ptr[regAux1]);
            }

            this->vpermd(vAux0, vPermMask, vAux0);
            if (isa == x64::avx2) {
                vPermMask.release();
            }

            this->uni_vmovups(this->ptr[this->regDst], vAux0);
        }

        // Main loop.
        this->L(lDstIdxLoop1);
        {
            this->add(this->regDst, this->getVecLen());
            this->sub(this->regWorkAmount, this->getDataElPerVec());
            this->cmp(this->regWorkAmount, this->getDataElPerVec());
            this->jl(lTail, this->T_NEAR);

            vBuff0 = calculateIdxShiftsForHalfIteration(shiftCalculator, vShufMask, true, std::move(vBuff0));
            vBuff1 = calculateIdxShiftsForHalfIteration(shiftCalculator, vShufMask, true, std::move(vBuff1));
            {
                RegistersPool::Reg<Vmm> vAux0{this->regPool};
                this->vshufps(vAux0, vBuff0, vBuff1, 0x88);

                if (isa == x64::avx2) {
                    vPermMask = RegistersPool::Reg<Vmm>{this->regPool};
                    RegistersPool::Reg<Xbyak::Reg64> regAux1{this->regPool};
                    this->mov(regAux1, reinterpret_cast<uintptr_t>(permMask8bitA2));
                    this->uni_vmovups(vPermMask, this->ptr[regAux1]);
                }
                this->vpermd(vAux0, vPermMask, vAux0);

                this->uni_vmovups(this->ptr[this->regDst], vAux0);
            }

            this->jmp(lDstIdxLoop1, this->T_NEAR);
        }
    }
    this->L(lTail);
    this->tail(shiftCalculator, true);
}

template<x64::cpu_isa_t isa>
void jitGatherKernelBase<isa>::generateForStaticShapes() {
    this->uploadParamPtrWithVpbroadcastd(this->vmmSpecIdxSizeB, GET_OFF(specIndicesSize));
    this->uni_vpslld(this->vmmSpecIdxSizeB, this->vmmSpecIdxSizeB, this->idxTypeShift); // multiply by indexes type size.

    this->uploadParamPtrWithVmovups(this->vmmSpecIdxB, GET_OFF(specIdxB));

    if (this->beforeAxisSize != 1lu) {
        this->uploadParamPtrWithVmovups(this->vmmSrcBeforeAxisSumB, GET_OFF(dataBeforeAxisSumB));
    }
    getShiftCalculator().allocateRegisters(*this);
    getShiftCalculator().uploadParamsForApproachSpecific(*this);
    this->process(getShiftCalculator());
}

template<x64::cpu_isa_t isa>
void jitGatherKernelBase<isa>::generateForDynamicShapes() {
    typename jitGatherKernelBase<isa>::template ShiftCalculatorImpl<ElementwiseCase, Long> elementwiseLong;
    typename jitGatherKernelBase<isa>::template ShiftCalculatorImpl<ElementwiseCase, Short> elementwiseShort;
    typename jitGatherKernelBase<isa>::template ShiftCalculatorImpl<BlockedCase, Short> blockedShort;
    elementwiseLong.allocateRegisters(*this);

    static const unsigned incVec[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    this->uploadParamPtrWithVpbroadcastd(this->vmmSpecIdxB, GET_OFF(start));
    {
        RegistersPool::Reg<Xbyak::Reg64> regAux1{this->regPool};
        this->mov(regAux1, reinterpret_cast<uintptr_t>(incVec));
        this->uni_vpaddd(this->vmmSpecIdxB, this->vmmSpecIdxB, this->ptr[regAux1]);
    }
    this->vcvtdq2ps(this->vmmSpecIdxB, this->vmmSpecIdxB);

    // Formula: specIndices = (start % specIndicesSize) * idxTypeSize
    this->uploadParamPtrWithVpbroadcastd(this->vmmSpecIdxSizeB, GET_OFF(specIndicesSize));
    {
        RegistersPool::Reg<Vmm> vAux1{this->regPool};
        this->uni_vcvtdq2ps(vAux1, this->vmmSpecIdxSizeB);
        this->uni_vdivps(this->vmmSrcBeforeAxisSumB, this->vmmSpecIdxB, vAux1);
        this->uni_vroundps(this->vmmSrcBeforeAxisSumB, this->vmmSrcBeforeAxisSumB, 0x1);
        this->uni_vfnmadd231ps(this->vmmSpecIdxB, this->vmmSrcBeforeAxisSumB, vAux1);
    }
    this->uni_vcvtps2dq(this->vmmSpecIdxB, this->vmmSpecIdxB);
    this->uni_vpslld(this->vmmSpecIdxB, this->vmmSpecIdxB, this->idxTypeShift); // multiply by indices type size.
    this->uni_vpslld(this->vmmSpecIdxSizeB, this->vmmSpecIdxSizeB, this->idxTypeShift); // multiply by indices type size.
    RegistersPool::Reg<Xbyak::Reg64>& regSpecIdxSizeB = elementwiseLong.regSpecIdxSizeB;
    this->uni_vmovd(Xbyak::Reg32(regSpecIdxSizeB.getIdx()), Xbyak::Xmm(this->vmmSpecIdxSizeB.getIdx()));

    {
        RegistersPool::Reg<Vmm> vAux1{this->regPool};
        this->uploadParamPtrWithVpbroadcastd(vAux1, GET_OFF(betweenBatchAndAxisSize));
        this->uni_vmovd(Xbyak::Reg32(elementwiseLong.regBetweenBatchAndAxisSize.getIdx()), Xbyak::Xmm(vAux1.getIdx()));
        this->uni_vcvtdq2ps(vAux1, vAux1);
        this->uni_vdivps(elementwiseLong.vmmIdxBatchSumB, this->vmmSrcBeforeAxisSumB, vAux1);
        this->uni_vroundps(elementwiseLong.vmmIdxBatchSumB, elementwiseLong.vmmIdxBatchSumB, 0x1);
        this->uni_vfnmadd231ps(this->vmmSrcBeforeAxisSumB, elementwiseLong.vmmIdxBatchSumB, vAux1);
    }
    this->uni_vcvtps2dq(this->vmmSrcBeforeAxisSumB, this->vmmSrcBeforeAxisSumB);
    this->uni_vmovd(Xbyak::Reg32(elementwiseLong.regBetweenBatchAndAxisIter.getIdx()), Xbyak::Xmm(this->vmmSrcBeforeAxisSumB.getIdx()));
    this->uni_vcvtps2dq(elementwiseLong.vmmIdxBatchSumB, elementwiseLong.vmmIdxBatchSumB);

    this->uploadParamPtrWithVpbroadcastd(elementwiseLong.vmmAxisAndAfterAxisSizeB, GET_OFF(axisAndAfterAxisSizeB));
    // Formula: srcBeforeAxisSum = ((start / specIndicesSize) % betweenBatchAndAxis) * axisAndAfterAxisSize + srcAfterBatchSize * idxBatchSum
    if (this->beforeAxisSize != 1lu) {
        RegistersPool::Reg<Vmm> vAux0{this->regPool};
        this->uni_vpmulld(this->vmmSrcBeforeAxisSumB, this->vmmSrcBeforeAxisSumB, elementwiseLong.vmmAxisAndAfterAxisSizeB);
        this->uploadParamPtrWithVpbroadcastd(vAux0, GET_OFF(srcAfterBatchSizeB));
        this->uni_vpmulld(vAux0, vAux0, elementwiseLong.vmmIdxBatchSumB);
        this->uni_vpaddd(this->vmmSrcBeforeAxisSumB, this->vmmSrcBeforeAxisSumB, vAux0);
    }

    // Formula: idxBatchSum = specIdxSize * (start / afterBatchSize)
    this->uni_vpmulld(elementwiseLong.vmmIdxBatchSumB, elementwiseLong.vmmIdxBatchSumB, this->vmmSpecIdxSizeB);

    Xbyak::Label lBlock, lEnd;
    const Xbyak::Reg64& regAux2 = this->rsi;
    this->mov(regAux2, this->ptr[this->regParams + GET_OFF(afterAxSize)]);
    this->cmp(regAux2, 1);
    this->jg(lBlock, this->T_NEAR);
    {
        Xbyak::Label lLessThanVector1, lTail1, lTail2, lE1;

        this->cmp(regSpecIdxSizeB, this->getVecLen());
        this->jl(lLessThanVector1, this->T_NEAR);
        {
            this->uni_vmovd(Xbyak::Reg32(elementwiseLong.regIdxIter.getIdx()), Xbyak::Xmm(this->vmmSpecIdxB.getIdx()));
            this->fillVlenVector(elementwiseLong.vmmVecLenB);

            this->process(elementwiseLong);
            elementwiseLong.releaseRegisters();
            this->jmp(lE1, this->T_NEAR);
        }
        this->L(lLessThanVector1);
        {
            elementwiseShort.allocateRegisters(*this);
            this->uploadParamPtrWithVmovups(elementwiseShort.vmmPermIdxMask, GET_OFF(permIdxMask));
            if (this->beforeAxisSize != 1lu) {
                this->uploadParamPtrWithVmovups(elementwiseShort.vmmBeforeAxDiffB, GET_OFF(beforeAxisDiff));
                if (this->getDataTypeSize() != 1)
                    this->uni_vpslld(elementwiseShort.vmmBeforeAxDiffB, elementwiseShort.vmmBeforeAxDiffB,
                                     this->getDataTypeShift()); // multiply by data type size
            }
            this->uploadParamPtrWithVpbroadcastd(elementwiseShort.vmmSrcAfterBatchSizeB, GET_OFF(srcAfterBatchSizeB));

            this->process(elementwiseShort);
            elementwiseShort.releaseRegisters();
        }
        this->L(lE1);
        this->jmp(lEnd, this->T_NEAR);
    }
    this->L(lBlock); {
        blockedShort.rSpecIdxAndAfterAxIterB = RegistersPool::Reg<Xbyak::Reg64>{this->regPool};
        blockedShort.rSpecIdxAndAfterAxSizeB = RegistersPool::Reg<Xbyak::Reg64>{this->regPool};

        blockedShort.vmmAxisAndAfterAxisSizeB = RegistersPool::Reg<Vmm>{this->regPool};
        blockedShort.vmmSrcAfterBatchSizeB = RegistersPool::Reg<Vmm>{this->regPool};
        blockedShort.vmmAfterAxisIdxB = RegistersPool::Reg<Vmm>{this->regPool};
        blockedShort.vmmAfterAxisPermMask = RegistersPool::Reg<Vmm>{this->regPool};

        Xbyak::Label lLessThanVector2, lTail3, lTail4, lE2;
        {
            RegistersPool::Reg<Vmm> vAux0{this->regPool};
            this->uploadParamPtrWithVpbroadcastd(blockedShort.vmmAfterAxisIdxB, GET_OFF(start));
            {
                RegistersPool::Reg<Xbyak::Reg64> regAux1{this->regPool};
                this->mov(regAux1, reinterpret_cast<uintptr_t>(incVec));
                this->uni_vpaddd(blockedShort.vmmAfterAxisIdxB, blockedShort.vmmAfterAxisIdxB, this->ptr[regAux1]);
                this->uni_vcvtdq2ps(blockedShort.vmmAfterAxisIdxB, blockedShort.vmmAfterAxisIdxB);

                // afterAxIdxB = (start % afterAxSize) * idxTypeSize
                this->movd(Xbyak::Xmm(vAux0.getIdx()), Xbyak::Reg32(regAux1.getIdx()));
            }
            RegistersPool::Reg<Vmm> vAux1{this->regPool};
            this->uni_vpbroadcastd(vAux1, Xbyak::Xmm(vAux0.getIdx()));
            this->uni_vcvtdq2ps(vAux1, vAux1);
            this->uni_vdivps(this->vmmSrcBeforeAxisSumB, blockedShort.vmmAfterAxisIdxB, vAux1);
            this->uni_vroundps(this->vmmSrcBeforeAxisSumB, this->vmmSrcBeforeAxisSumB, 0x1);
            this->uni_vfnmadd231ps(blockedShort.vmmAfterAxisIdxB, this->vmmSrcBeforeAxisSumB, vAux1);
            this->uni_vcvtps2dq(blockedShort.vmmAfterAxisIdxB, blockedShort.vmmAfterAxisIdxB);
            this->uni_vpslld(blockedShort.vmmAfterAxisIdxB, blockedShort.vmmAfterAxisIdxB,
                             this->idxTypeShift); // multiply by indices type size.

            this->cmp(regAux2, this->getDataElPerVec());
            this->jl(lLessThanVector2, this->T_NEAR);
            this->uni_vmovd(Xbyak::Reg32(blockedShort.rSpecIdxAndAfterAxIterB.getIdx()), Xbyak::Xmm(this->vmmSpecIdxB.getIdx()));
            //this->fillVlenVector(elementwiseLong.vmmVecLenB);

            // process(blockedLong); // not implemented
            this->jmp(lE2, this->T_NEAR);
            this->L(lLessThanVector2);
            RegistersPool::Reg<Vmm> vAux2{this->regPool};
            // Calculate permute mask
            this->uni_vmovd(Xbyak::Xmm(vAux0.getIdx()), Xbyak::Reg32(regAux2.getIdx()));
            this->uni_vpbroadcastd(vAux1, Xbyak::Xmm(vAux0.getIdx()));
            idxElPerVec = this->getIdxElPerVec();
            {
                RegistersPool::Reg<Xbyak::Reg64> regAux1{this->regPool};
                this->mov(regAux1, reinterpret_cast<uintptr_t>(&idxElPerVec));
                this->uni_vpbroadcastd(vAux0, this->ptr[regAux1]);
                this->uni_vpsubd(blockedShort.vmmAfterAxisPermMask, vAux0, vAux1);
                this->mov(regAux1, reinterpret_cast<uintptr_t>(incVec));
                this->uni_vpaddd(blockedShort.vmmAfterAxisPermMask, blockedShort.vmmAfterAxisPermMask,
                                 this->ptr[regAux1]);
            }
            for (int i = 0; i < 6; i++) {
                if (isa == x64::avx512_core) {
                    Xbyak::Opmask kMask2 = Xbyak::Opmask(vAux2.getIdx());
                    this->vpcmpgtd(kMask2, vAux0, blockedShort.vmmAfterAxisPermMask);
                    this->uni_vpsubd(static_cast<Vmm &>(blockedShort.vmmAfterAxisPermMask) | kMask2,
                                     blockedShort.vmmAfterAxisPermMask, vAux1);
                } else {
                    this->vpcmpgtd(vAux2, vAux0, blockedShort.vmmAfterAxisPermMask);
                    this->vpandn(vAux2, vAux2, vAux1);
                    this->uni_vpsubd(blockedShort.vmmAfterAxisPermMask, blockedShort.vmmAfterAxisPermMask, vAux2);
                }
            }
        }

        this->process(blockedShort);
        this->L(lE2);
    }
    this->L(lEnd);
}

template<x64::cpu_isa_t isa> template<typename unused>
void jitGatherKernelBase<isa>::ShiftCalculatorImpl<ElementwiseCase, Short, unused>::allocateRegisters(jitGatherKernelBase& kernel) {
    vmmBeforeAxDiffB = RegistersPool::Reg<Vmm>{kernel.regPool, 12};
    vmmSrcAfterBatchSizeB = RegistersPool::Reg<Vmm>{kernel.regPool, 13};
    vmmPermIdxMask = RegistersPool::Reg<Vmm>{kernel.regPool, 14};
}

template<x64::cpu_isa_t isa> template<typename unused>
void jitGatherKernelBase<isa>::ShiftCalculatorImpl<ElementwiseCase, Short, unused>::releaseRegisters() {
    vmmBeforeAxDiffB.release();
    vmmSrcAfterBatchSizeB.release();
    vmmPermIdxMask.release();
}

template<x64::cpu_isa_t isa> template<typename unused>
void jitGatherKernelBase<isa>::ShiftCalculatorImpl<ElementwiseCase, Short, unused>::uploadParamsForApproachSpecific(
        jitGatherKernelBase& kernel) {
    if (kernel.specIdxSize != 1 && kernel.specIdxSize != 2 && kernel.specIdxSize != 4 && kernel.specIdxSize != 8 &&
        kernel.specIdxSize != 16) {
        kernel.uploadParamPtrWithVmovups(vmmPermIdxMask, GET_OFF(permIdxMask));
    }
    if (kernel.beforeAxisSize != 1lu) {
        kernel.uploadParamPtrWithVmovups(vmmBeforeAxDiffB, GET_OFF(beforeAxisDiff));
        if (kernel.getDataTypeSize() != 1)
            kernel.uni_vpslld(vmmBeforeAxDiffB, vmmBeforeAxDiffB, kernel.getDataTypeShift()); // multiply by data type size
    }
    if (kernel.batchDims > 0lu) {
        kernel.uploadParamPtrWithVpbroadcastd(vmmSrcAfterBatchSizeB, GET_OFF(srcAfterBatchSizeB));
    }
}

template <>
void jitGatherKernelBase<x64::avx512_core>::fillVlenVector(RegistersPool::Reg<Vmm>& vmmVecLenB) {
    RegistersPool::Reg<Xbyak::Reg32> reg32Aux1 {regPool};
    mov(reg32Aux1, this->getVecLen());
    vpbroadcastd(vmmVecLenB, reg32Aux1);
}
template <>
void jitGatherKernelBase<x64::avx2>::fillVlenVector(RegistersPool::Reg<Vmm>& vmmVecLenB) {
    vpcmpeqd(vmmVecLenB, vmmVecLenB, vmmVecLenB);
    vpsrld(vmmVecLenB, vmmVecLenB, 31); // Right shift to 1.
    uni_vpslld(vmmVecLenB, vmmVecLenB, 5);  // Left shift to 32.
}

template<x64::cpu_isa_t isa> template<typename unused>
void jitGatherKernelBase<isa>::ShiftCalculatorImpl<ElementwiseCase, Long, unused>::allocateRegisters(jitGatherKernelBase& kernel) {
    regBetweenBatchAndAxisSize = Xbyak::Reg64(kernel.rbx.getIdx());
    regSpecIdxSizeB = RegistersPool::Reg<Xbyak::Reg64>{kernel.regPool, 13};
    regBetweenBatchAndAxisIter = RegistersPool::Reg<Xbyak::Reg64>{kernel.regPool, 15};
    regIdxIter = RegistersPool::Reg<Xbyak::Reg64>{kernel.regPool, 11};

    vmmIdxBatchSumB = RegistersPool::Reg<Vmm>{kernel.regPool, 14};
    vmmVecLenB = RegistersPool::Reg<Vmm>{kernel.regPool, 13};
    vmmAxisAndAfterAxisSizeB = RegistersPool::Reg<Vmm>{kernel.regPool, 12};
}

template<x64::cpu_isa_t isa> template<typename unused>
void jitGatherKernelBase<isa>::ShiftCalculatorImpl<ElementwiseCase, Long, unused>::releaseRegisters() {
    regSpecIdxSizeB.release();
    regBetweenBatchAndAxisIter.release();
    regIdxIter.release();

    vmmIdxBatchSumB.release();
    vmmVecLenB.release();
    vmmAxisAndAfterAxisSizeB.release();
}

template<x64::cpu_isa_t isa> template<typename unused>
void jitGatherKernelBase<isa>::ShiftCalculatorImpl<ElementwiseCase, Long, unused>::uploadParamsForApproachSpecific(
        jitGatherKernelBase& kernel) {
    kernel.uni_vmovd(Xbyak::Reg32(regSpecIdxSizeB.getIdx()), Xbyak::Xmm(kernel.vmmSpecIdxSizeB.getIdx()));
    if (kernel.beforeAxisSize != 1lu) {
        kernel.uploadParamPtrWithVpbroadcastd(vmmAxisAndAfterAxisSizeB, GET_OFF(axisAndAfterAxisSizeB));
    }
    kernel.uploadParamPtrWithVmovups(vmmIdxBatchSumB, GET_OFF(idxBatchSumB));
    RegistersPool::Reg<Xbyak::Reg64> regAux {kernel.regPool};
    kernel.mov(regAux, kernel.ptr[kernel.regParams + GET_OFF(betweenBatchAndAxisSize)]);
    kernel.mov(regBetweenBatchAndAxisSize, kernel.ptr[regAux]);
    kernel.mov(regBetweenBatchAndAxisIter, kernel.ptr[kernel.regParams + GET_OFF(betweenBatchAndAxisIter)]);

    kernel.uni_vmovd(Xbyak::Reg32(regIdxIter.getIdx()), Xbyak::Xmm(kernel.vmmSpecIdxB.getIdx()));
    kernel.fillVlenVector(vmmVecLenB);
}

template<x64::cpu_isa_t isa> template<typename unused>
void jitGatherKernelBase<isa>::ShiftCalculatorImpl<BlockedCase, Short, unused>::allocateRegisters(jitGatherKernelBase& kernel) {
    rSpecIdxAndAfterAxIterB = RegistersPool::Reg<Xbyak::Reg64>{kernel.regPool, 11};
    rSpecIdxAndAfterAxSizeB = RegistersPool::Reg<Xbyak::Reg64>{kernel.regPool, 13};

    vmmAxisAndAfterAxisSizeB = RegistersPool::Reg<Vmm>{kernel.regPool, 12};
    vmmSrcAfterBatchSizeB = RegistersPool::Reg<Vmm>{kernel.regPool, 13};
    vmmAfterAxisIdxB = RegistersPool::Reg<Vmm>{kernel.regPool, 15};
    vmmAfterAxisPermMask = RegistersPool::Reg<Vmm>{kernel.regPool, 14};
    vmmSpecIdxDiff = RegistersPool::Reg<Vmm>{kernel.regPool, 4};
    vmmAfterAxisSize = RegistersPool::Reg<Vmm>{kernel.regPool, 5};
    vmmBeforeAxPermMask = RegistersPool::Reg<Vmm>{kernel.regPool, 6};
}

template<x64::cpu_isa_t isa> template<typename unused>
void jitGatherKernelBase<isa>::ShiftCalculatorImpl<BlockedCase, Short, unused>::uploadParamsForApproachSpecific(
        jitGatherKernelBase& kernel) {
    kernel.uploadParamPtrWithVmovups(vmmAfterAxisIdxB, GET_OFF(afterAxIdxB));
    kernel.uploadParamPtrWithVmovups(vmmAfterAxisPermMask, GET_OFF(afterAxisPermMask));
    kernel.uploadParamPtrWithVmovups(vmmSpecIdxDiff, GET_OFF(specIdxDiff));
    kernel.uploadParamPtrWithVpbroadcastd(vmmSrcAfterBatchSizeB, GET_OFF(srcAfterBatchSizeB));
    kernel.uploadParamPtrWithVpbroadcastd(vmmAfterAxisSize, GET_OFF(afterAxisSize));

    if (kernel.beforeAxisSize != 1lu) {
        kernel.mov(rSpecIdxAndAfterAxIterB, kernel.ptr[kernel.regParams + GET_OFF(specIdxAndAfterAxIterB)]);
        kernel.mov(rSpecIdxAndAfterAxSizeB, kernel.ptr[kernel.regParams + GET_OFF(specIdxAndAfterAxSizeB)]);
        if (kernel.specIdxSize * kernel.afterAxisSize < kernel.getIdxElPerVec()) {
            auto& vmmBeforeAxDiffB = vmmAxisAndAfterAxisSizeB;
            kernel.uploadParamPtrWithVmovups(vmmBeforeAxDiffB, GET_OFF(beforeAxisDiff));
        } else {
            kernel.uploadParamPtrWithVpbroadcastd(vmmAxisAndAfterAxisSizeB, GET_OFF(axisAndAfterAxisSizeB));
        }
        const uint64_t specIdxAndAfterAxisSize = kernel.specIdxSize * kernel.afterAxisSize;
        if (specIdxAndAfterAxisSize != 1 && specIdxAndAfterAxisSize != 2 && specIdxAndAfterAxisSize != 4 &&
            specIdxAndAfterAxisSize != 8 && specIdxAndAfterAxisSize != 16) {
            kernel.uploadParamPtrWithVmovups(vmmBeforeAxPermMask, GET_OFF(beforeAxisPermMask));
        }
    }
}


template<x64::cpu_isa_t isa>
poolVmask<isa> jitGatherKernelBase<isa>::calcAllOnesMask(Vmm& vAux) {
    RegistersPool::Reg<Vmask> onesMask {regPool};
    vpcmpeqd(onesMask, vAux, vAux);
    return onesMask;
}

template<x64::cpu_isa_t isa>
std::tuple<poolVmask<isa> /*kDstMask*/, poolVmm<isa> /*vDstShifts*/>
        jitGatherKernelBase<isa>::calculateIndexesForShortCase(Vmm& srcBeforeAxisSumB, Vmm& srcAfterBatchSizeB) {
    RegistersPool::Reg<Vmm> vDstShifts;
    if (batchDims > 0lu) {
        // Calculate indices batch sum.
        RegistersPool::Reg<Vmm> vAux0{regPool};
        uni_vcvtdq2ps(vAux0, srcBeforeAxisSumB);
        {
            RegistersPool::Reg<Vmm> vmmSrcAfterBatchSizeBFloat{regPool};
            uni_vcvtdq2ps(vmmSrcAfterBatchSizeBFloat, srcAfterBatchSizeB);
            uni_vdivps(vAux0, vAux0, vmmSrcAfterBatchSizeBFloat);
        }
        uni_vroundps(vAux0, vAux0, 0x1);
        uni_vcvtps2dq(vAux0, vAux0);

        uni_vpmulld(vAux0, vAux0, vmmSpecIdxSizeB);
        uni_vpaddd(vAux0, vAux0, vmmSpecIdxB);

        vDstShifts = RegistersPool::Reg<Vmm> {regPool};
        uniVpGatherDd(vDstShifts, ptr[regIndices + vAux0], calcAllOnesMask(vAux0));
    } else {
        vDstShifts = RegistersPool::Reg<Vmm> {regPool};
        uniVpGatherDd(vDstShifts, ptr[regIndices + vmmSpecIdxB], calcAllOnesMask(vDstShifts));
    }

    RegistersPool::Reg<Vmask> kDstMask = normalizeIndexesAndCalcShifts(vDstShifts);
    return {std::move(kDstMask), std::move(vDstShifts)};
}

template<x64::cpu_isa_t isa> template<typename unused>
std::tuple<poolVmask<isa> /*kDstMask*/, poolVmm<isa> /*vDstShifts*/>
jitGatherKernelBase<isa>::ShiftCalculatorImpl<ElementwiseCase, Short, unused>::calcSrcShift(
        jitGatherKernelBase& kernel, bool shiftFirst) {
    if (shiftFirst) {
        if (kernel.beforeAxisSize != 1lu)
            kernel.uni_vpaddd(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB, vmmBeforeAxDiffB);
        // No sense to permute if specIdxSize is one of {1, 2, 4, 8, 16}. 0 is reserved for dynamic case.
        if (kernel.specIdxSize != 1 && kernel.specIdxSize != 2 && kernel.specIdxSize != 4 && kernel.specIdxSize != 8 &&
            kernel.specIdxSize != 16) {
            kernel.vpermd(kernel.vmmSpecIdxB, vmmPermIdxMask, kernel.vmmSpecIdxB);
            if (kernel.beforeAxisSize != 1lu)
                kernel.vpermd(vmmBeforeAxDiffB, vmmPermIdxMask, vmmBeforeAxDiffB);
        }
    }

    RegistersPool::Reg<Vmm> vDstShifts;
    RegistersPool::Reg<Vmask> kDstMask;
    std::tie(kDstMask, vDstShifts) = kernel.calculateIndexesForShortCase(kernel.vmmSrcBeforeAxisSumB, vmmSrcAfterBatchSizeB);

    if (kernel.beforeAxisSize != 1lu)
        kernel.uni_vpaddd(vDstShifts, vDstShifts, kernel.vmmSrcBeforeAxisSumB);
    return {std::move(kDstMask), std::move(vDstShifts)};
}

template <>
void jitGatherKernelBase<x64::avx2>::normWithUpperBound(Vmm& vTarget, Vmm& vMax, Vmask& kAuxMask) {
    vpcmpgtd(kAuxMask, vMax, vTarget);
    vpandn(kAuxMask, kAuxMask, vMax);
    uni_vpsubd(vTarget, vTarget, kAuxMask);
}

template <>
void jitGatherKernelBase<x64::avx512_core>::normWithUpperBound(Vmm& vTarget, Vmm& vMax, Vmask& kAuxMask) {
    vpcmpd(kAuxMask, vMax, vTarget, 2); // 2 -> LE
    uni_vpsubd(vTarget | kAuxMask, vTarget, vMax);
}

template<x64::cpu_isa_t isa> template<typename unused>
std::tuple<poolVmask<isa> /*kDstMask*/, poolVmm<isa> /*vDstShifts*/>
jitGatherKernelBase<isa>::ShiftCalculatorImpl<BlockedCase, Short, unused>::calcSrcShift(
        jitGatherKernelBase& kernel, bool shiftFirst) {
    RegistersPool::Reg<Vmm> vAux1{kernel.regPool};
    const uint64_t specIdxAndAfterAxisSize = kernel.specIdxSize * kernel.afterAxisSize;

    if (shiftFirst) {
        if (kernel.specIdxSize != 1 && vmmSpecIdxDiff.isInitialized()) {
            kernel.uni_vpaddd(kernel.vmmSpecIdxB, kernel.vmmSpecIdxB, vmmSpecIdxDiff);
            RegistersPool::Reg<Vmask> kAuxMask0 {kernel.regPool};
            kernel.normWithUpperBound(kernel.vmmSpecIdxB, kernel.vmmSpecIdxSizeB, kAuxMask0);
        }
        // No sense to permute if afterAxisSize is one of {1, 2, 4, 8, 16}. 0 is reserved for dynamic case.
        if (kernel.afterAxisSize != 1 && kernel.afterAxisSize != 2 && kernel.afterAxisSize != 4 &&
            kernel.afterAxisSize != 8 && kernel.afterAxisSize != 16) {
            kernel.vpermd(vmmAfterAxisIdxB, vmmAfterAxisPermMask, vmmAfterAxisIdxB);
            if (kernel.specIdxSize != 1 && vmmSpecIdxDiff.isInitialized())
                kernel.vpermd(vmmSpecIdxDiff, vmmAfterAxisPermMask, vmmSpecIdxDiff);
        }

        if (kernel.beforeAxisSize != 1lu) {
            if (!kernel.dynamicShapes) {
                if (specIdxAndAfterAxisSize > 0lu && specIdxAndAfterAxisSize <= kernel.getIdxElPerVec()) {
                    auto& vmmBeforeAxDiffB = vmmAxisAndAfterAxisSizeB;
                    kernel.uni_vpaddd(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB, vmmBeforeAxDiffB);
                    kernel.uni_vmovups(vAux1, kernel.vmmSrcBeforeAxisSumB);
                    if (specIdxAndAfterAxisSize != 1 && specIdxAndAfterAxisSize != 2 &&
                        specIdxAndAfterAxisSize != 4 &&
                        specIdxAndAfterAxisSize != 8 && specIdxAndAfterAxisSize != 16 && vmmBeforeAxPermMask.isInitialized())
                        kernel.vpermd(vmmBeforeAxDiffB, vmmBeforeAxPermMask, vmmBeforeAxDiffB);
                } else {
                    Xbyak::Label lBeforeAxStep, lBeforeAxStepEnd;
                    kernel.add(rSpecIdxAndAfterAxIterB, kernel.getIdxElPerVec() * kernel.getDataTypeSize());
                    kernel.cmp(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);
                    kernel.jl(lBeforeAxStep, kernel.T_NEAR);
                    kernel.sub(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);

                    RegistersPool::Reg<Vmm> vAux0{kernel.regPool};
                    if (vmmAfterAxisSize.isInitialized()) {
                        kernel.vpmulld(vAux0, kernel.vmmSpecIdxB, vmmAfterAxisSize);
                    }
                    kernel.uni_vpaddd(vAux0, vAux0, vmmAfterAxisIdxB);
                    kernel.uni_vpbroadcastd(vAux1, Xbyak::Xmm(vAux0.getIdx()));
                    if (isa == x64::avx512_core) {
                        RegistersPool::Reg<Xbyak::Opmask> kMask0 {kernel.regPool};
                        kernel.vpcmpgtd(kMask0, vAux1, vAux0);
                        kernel.uni_vmovups(vAux1, kernel.vmmSrcBeforeAxisSumB);
                        kernel.uni_vpaddd(static_cast<Vmm&>(vAux1) | kMask0, kernel.vmmSrcBeforeAxisSumB,
                                          vmmAxisAndAfterAxisSizeB);
                    } else {
                        kernel.vpcmpgtd(vAux1, vAux1, vAux0);
                        kernel.vpand(vAux1, vAux1, vmmAxisAndAfterAxisSizeB);
                        kernel.uni_vpaddd(vAux1, kernel.vmmSrcBeforeAxisSumB, vAux1);
                    }
                    kernel.uni_vpaddd(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB,
                                      vmmAxisAndAfterAxisSizeB);
                    kernel.jmp(lBeforeAxStepEnd);
                    kernel.L(lBeforeAxStep);
                    kernel.uni_vmovups(vAux1, kernel.vmmSrcBeforeAxisSumB);
                    kernel.L(lBeforeAxStepEnd);
                }
            } else {
            }
        }
    } else {
        if (kernel.beforeAxisSize != 1lu) {
            kernel.uni_vmovups(vAux1, kernel.vmmSrcBeforeAxisSumB);
            if (specIdxAndAfterAxisSize > kernel.getIdxElPerVec()) {
                // Broadcast the last element.
                if (isa == x64::avx512_core) {
                    kernel.vshuff64x2(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB,
                                      kernel.vmmSrcBeforeAxisSumB, 0xFF);
                } else {
                    kernel.vpermq(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB, 0xFF);
                }
                kernel.vpshufd(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB, 0xFF);

                Xbyak::Label lBeforeAxStepEnd1;
                kernel.add(rSpecIdxAndAfterAxIterB, kernel.getIdxElPerVec() * kernel.getDataTypeSize());
                kernel.cmp(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);
                kernel.jl(lBeforeAxStepEnd1, kernel.T_NEAR);
                kernel.sub(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);
                kernel.cmp(rSpecIdxAndAfterAxIterB, 0);
                kernel.jne(lBeforeAxStepEnd1, kernel.T_NEAR);
                kernel.uni_vpaddd(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB,
                                  vmmAxisAndAfterAxisSizeB);
                kernel.L(lBeforeAxStepEnd1);
            }
        }
    }

    RegistersPool::Reg<Vmm> vDstShifts;
    RegistersPool::Reg<Vmask> kDstMask;
    std::tie(kDstMask, vDstShifts) = kernel.calculateIndexesForShortCase(vAux1, vmmSrcAfterBatchSizeB);

    if (kernel.afterAxisSize != 1lu) {
        if (vmmAfterAxisSize.isInitialized()) {
            kernel.vpmulld(vDstShifts, vDstShifts, vmmAfterAxisSize);
        }
        kernel.uni_vpaddd(vDstShifts, vDstShifts, vmmAfterAxisIdxB);
    }
    if (kernel.beforeAxisSize != 1lu)
        kernel.uni_vpaddd(vDstShifts, vDstShifts, vAux1);
    return {std::move(kDstMask), std::move(vDstShifts)};
}

template<x64::cpu_isa_t isa> template<typename unused>
std::tuple<poolVmask<isa> /*kDstMask*/, poolVmm<isa> /*vDstShifts*/>
jitGatherKernelBase<isa>::ShiftCalculatorImpl<ElementwiseCase, Long, unused>::calcSrcShift(
        jitGatherKernelBase& kernel, bool shiftFirst) {
    if (isa == x64::avx2) {
        Xbyak::Label lIdxStride, lExit;
        if (shiftFirst)
            kernel.uni_vpaddd(kernel.vmmSpecIdxB, kernel.vmmSpecIdxB, vmmVecLenB);

        kernel.add(regIdxIter, kernel.getVecLen());
        kernel.cmp(regIdxIter, regSpecIdxSizeB);
        kernel.jge(lIdxStride, kernel.T_NEAR);
        RegistersPool::Reg<Vmask> kDstMask;
        RegistersPool::Reg<Vmm> vDstShifts {kernel.regPool};
        {
            RegistersPool::Reg<Xbyak::Reg32> reg32Aux1{kernel.regPool};
            if (kernel.batchDims > 0lu) {
                kernel.uni_vpaddd(vDstShifts, vmmIdxBatchSumB, kernel.vmmSpecIdxB);
                kernel.uni_vmovd(reg32Aux1, Xbyak::Xmm(vDstShifts.getIdx()));
            } else {
                kernel.uni_vmovd(reg32Aux1, Xbyak::Xmm(kernel.vmmSpecIdxB.getIdx()));
            }
            kernel.vmovdqu(vDstShifts, kernel.ptr[kernel.regIndices + Xbyak::Reg64(reg32Aux1.getIdx())]);
            kDstMask = kernel.normalizeIndexesAndCalcShifts(vDstShifts);
            if (kernel.beforeAxisSize != 1lu)
                kernel.uni_vpaddd(vDstShifts, vDstShifts, kernel.vmmSrcBeforeAxisSumB);
        }
        kernel.jmp(lExit, kernel.T_NEAR);
        kernel.L(lIdxStride);
        {
            kernel.sub(regIdxIter, regSpecIdxSizeB);
            RegistersPool::Reg<Vmm> vAux0;
            RegistersPool::Reg<Vmm> vAux1{kernel.regPool};
            kernel.vpcmpeqd(kDstMask, vAux1, vAux1);
            if (shiftFirst) {
                vAux0 = RegistersPool::Reg<Vmm>{kernel.regPool};
                kernel.vpcmpgtd(vAux0, kernel.vmmSpecIdxSizeB, kernel.vmmSpecIdxB);
                kernel.vpandn(vAux1, vAux0, kernel.vmmSpecIdxSizeB);
                kernel.uni_vpsubd(vAux1, kernel.vmmSpecIdxB, vAux1);
                if (kernel.batchDims > 0lu)
                    kernel.uni_vpaddd(vAux1, vmmIdxBatchSumB, vAux1);
                kernel.uni_vpsubd(kernel.vmmSpecIdxB, kernel.vmmSpecIdxB, kernel.vmmSpecIdxSizeB);
            } else {
                if (kernel.batchDims > 0lu) {
                    RegistersPool::Reg<Vmm> vAux{kernel.regPool};
                    kernel.uni_vpaddd(vAux, vmmIdxBatchSumB, kernel.vmmSpecIdxB);
                    kernel.uniVpGatherDd(vDstShifts, kernel.ptr[kernel.regIndices + vAux], kDstMask);
                } else {
                    kernel.uniVpGatherDd(vDstShifts, kernel.ptr[kernel.regIndices + kernel.vmmSpecIdxB], kDstMask);
                }
                kDstMask = kernel.normalizeIndexesAndCalcShifts(vDstShifts, std::move(kDstMask));

                vAux0 = RegistersPool::Reg<Vmm>{kernel.regPool};
                kernel.uni_vpbroadcastd(vAux0, Xbyak::Xmm(kernel.vmmSpecIdxB.getIdx()));
                kernel.vpcmpgtd(vAux1, vAux0, kernel.vmmSpecIdxB);
                kernel.vpandn(vAux0, vAux1, kernel.vmmSpecIdxSizeB);
                kernel.uni_vpsubd(kernel.vmmSpecIdxB, kernel.vmmSpecIdxB, vAux0);

                if (kernel.beforeAxisSize != 1lu) {
                    kernel.uni_vpaddd(vDstShifts, vDstShifts, kernel.vmmSrcBeforeAxisSumB);
                    kernel.vpandn(vAux0, vAux1, vmmAxisAndAfterAxisSizeB);
                    kernel.uni_vpaddd(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB, vAux0);
                }
            }

            if (kernel.batchDims > 0lu) {
                Xbyak::Label l1;
                kernel.inc(regBetweenBatchAndAxisIter);
                kernel.cmp(regBetweenBatchAndAxisIter, regBetweenBatchAndAxisSize);
                kernel.jl(l1, kernel.T_NEAR);
                kernel.mov(regBetweenBatchAndAxisIter, 0);
                if (shiftFirst) {
                    kernel.uni_vpaddd(vmmIdxBatchSumB, vmmIdxBatchSumB, kernel.vmmSpecIdxSizeB);
                    kernel.vpandn(vDstShifts, vAux0, kernel.vmmSpecIdxSizeB);
                    kernel.uni_vpaddd(vAux1, vAux1, vDstShifts);
                } else {
                    kernel.vpandn(vAux0, vAux1, kernel.vmmSpecIdxSizeB);
                    kernel.uni_vpaddd(vmmIdxBatchSumB, vmmIdxBatchSumB, vAux0);
                }
                kernel.L(l1);
            }

            if (shiftFirst) {
                kernel.uniVpGatherDd(vDstShifts, kernel.ptr[kernel.regIndices + vAux1], kDstMask);
                vAux1.release();
                kDstMask = kernel.normalizeIndexesAndCalcShifts(vDstShifts, std::move(kDstMask));

                if (kernel.beforeAxisSize != 1lu) {
                    kernel.vpandn(vAux0, vAux0, vmmAxisAndAfterAxisSizeB);
                    kernel.uni_vpaddd(vAux0, vAux0, kernel.vmmSrcBeforeAxisSumB);
                    kernel.uni_vpaddd(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB,
                                      vmmAxisAndAfterAxisSizeB);

                    kernel.uni_vpaddd(vDstShifts, vDstShifts, vAux0);
                }
            }
        }
        kernel.L(lExit);
        return {std::move(kDstMask), std::move(vDstShifts)};
    } else if (isa == x64::avx512_core) {
        RegistersPool::Reg<Xbyak::Zmm> vAux0{kernel.regPool};
        RegistersPool::Reg<Xbyak::Zmm> vAux1{kernel.regPool};
        RegistersPool::Reg<Xbyak::Opmask> kAuxMask1{kernel.regPool};

        Xbyak::Label lIdxStride, lExit;
        if (shiftFirst)
            kernel.uni_vpaddd(kernel.vmmSpecIdxB, kernel.vmmSpecIdxB, vmmVecLenB);

        kernel.add(regIdxIter, kernel.getVecLen());
        kernel.cmp(regIdxIter, regSpecIdxSizeB);
        kernel.jge(lIdxStride, kernel.T_NEAR);
        RegistersPool::Reg<Xbyak::Reg32> reg32Aux1 {kernel.regPool};
        RegistersPool::Reg<Vmm> vDstShifts {kernel.regPool};
        RegistersPool::Reg<Vmask> kDstMask;
        {
            if (kernel.batchDims > 0lu) {
                kernel.uni_vpaddd(vDstShifts, vmmIdxBatchSumB, kernel.vmmSpecIdxB);
                kernel.uni_vmovd(reg32Aux1, Xbyak::Xmm(vDstShifts.getIdx()));
            } else {
                kernel.uni_vmovd(reg32Aux1, Xbyak::Xmm(kernel.vmmSpecIdxB.getIdx()));
            }
            kernel.vmovdqu64(vDstShifts, kernel.ptr[kernel.regIndices + Xbyak::Reg64(reg32Aux1.getIdx())]);
            kDstMask = kernel.normalizeIndexesAndCalcShifts(vDstShifts);
            if (kernel.beforeAxisSize != 1lu)
                kernel.uni_vpaddd(vDstShifts, vDstShifts, kernel.vmmSrcBeforeAxisSumB);
        }
        kernel.jmp(lExit, kernel.T_NEAR);
        kernel.L(lIdxStride);
        {
            kernel.sub(regIdxIter, regSpecIdxSizeB);
            kernel.vpcmpeqd(kDstMask, vDstShifts, vDstShifts);
            if (shiftFirst) {
                kernel.vpcmpd(kAuxMask1, kernel.vmmSpecIdxSizeB, kernel.vmmSpecIdxB, 2); // 2 -> LE
                if (kernel.batchDims > 0lu) {
                    kernel.uni_vpaddd(vAux1, vmmIdxBatchSumB, kernel.vmmSpecIdxB);
                    kernel.uni_vpsubd(static_cast<Xbyak::Zmm&>(vAux1) | kAuxMask1, vAux1, kernel.vmmSpecIdxSizeB);
                } else {
                    kernel.uni_vmovups(vAux1, kernel.vmmSpecIdxB);
                    kernel.uni_vpsubd(static_cast<Xbyak::Zmm&>(vAux1) | kAuxMask1, kernel.vmmSpecIdxB,
                                      kernel.vmmSpecIdxSizeB);
                }
                kernel.uni_vpsubd(kernel.vmmSpecIdxB, kernel.vmmSpecIdxB, kernel.vmmSpecIdxSizeB);
            } else {
                if (kernel.batchDims > 0lu) {
                    kernel.uni_vpaddd(vAux0, vmmIdxBatchSumB, kernel.vmmSpecIdxB);
                    kernel.uniVpGatherDd(vDstShifts, kernel.ptr[kernel.regIndices + vAux0], kDstMask);
                } else {
                    kernel.uniVpGatherDd(vDstShifts, kernel.ptr[kernel.regIndices + kernel.vmmSpecIdxB], kDstMask);
                }
                kDstMask = kernel.normalizeIndexesAndCalcShifts(vDstShifts, std::move(kDstMask));

                kernel.uni_vpbroadcastd(vAux0, Xbyak::Xmm(kernel.vmmSpecIdxB.getIdx()));
                kernel.vpcmpd(kAuxMask1, vAux0, kernel.vmmSpecIdxB, 2); // 2 -> LE
                kernel.uni_vpsubd(Xbyak::Zmm(kernel.vmmSpecIdxB.getIdx()) | kAuxMask1, kernel.vmmSpecIdxB,
                                  kernel.vmmSpecIdxSizeB);

                if (kernel.beforeAxisSize != 1lu) {
                    kernel.uni_vpaddd(vDstShifts, vDstShifts, kernel.vmmSrcBeforeAxisSumB);
                    kernel.uni_vpaddd(Xbyak::Zmm(kernel.vmmSrcBeforeAxisSumB.getIdx()) | kAuxMask1,
                                      kernel.vmmSrcBeforeAxisSumB, vmmAxisAndAfterAxisSizeB);
                }
            }

            if (kernel.batchDims > 0lu) {
                Xbyak::Label l1;
                kernel.inc(regBetweenBatchAndAxisIter);
                kernel.cmp(regBetweenBatchAndAxisIter, regBetweenBatchAndAxisSize);
                kernel.jl(l1, kernel.T_NEAR);
                kernel.mov(regBetweenBatchAndAxisIter, 0);
                if (shiftFirst) {
                    kernel.uni_vpaddd(vmmIdxBatchSumB, vmmIdxBatchSumB, kernel.vmmSpecIdxSizeB);
                    kernel.uni_vpaddd(static_cast<Xbyak::Zmm&>(vAux1) | kAuxMask1, vAux1, kernel.vmmSpecIdxSizeB);
                } else {
                    kernel.uni_vpaddd(Xbyak::Zmm(vmmIdxBatchSumB.getIdx()) | kAuxMask1, vmmIdxBatchSumB,
                                      kernel.vmmSpecIdxSizeB);
                }
                kernel.L(l1);
            }

            if (shiftFirst) {
                kernel.uniVpGatherDd(vDstShifts, kernel.ptr[kernel.regIndices + vAux1], kDstMask);
                kDstMask = kernel.normalizeIndexesAndCalcShifts(vDstShifts, std::move(kDstMask));

                if (kernel.beforeAxisSize != 1lu) {
                    kernel.uni_vpaddd(vDstShifts, vDstShifts, kernel.vmmSrcBeforeAxisSumB);
                    kernel.uni_vpaddd(static_cast<Vmm>(vDstShifts) | kAuxMask1, vDstShifts, vmmAxisAndAfterAxisSizeB);
                    kernel.uni_vpaddd(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB,
                                      vmmAxisAndAfterAxisSizeB);
                }
            }
        }
        kernel.L(lExit);
        return {std::move(kDstMask), std::move(vDstShifts)};
    }
}


template <x64::cpu_isa_t isa, DataTypeSize S, AfterAxisCase C>
std::shared_ptr<jitGatherKernelInterface> createJitUniGatherKernel(bool isShortCase) {
    if (isShortCase)
        return std::make_shared<jitGatherKernelForStaticShapes<isa, S, C, Short>>();
    if (C == ElementwiseCase)
        return std::make_shared<jitGatherKernelForStaticShapes<isa, S, ElementwiseCase, Long>>();
    return {}; // BlockedCase Long not implemented
}

template <x64::cpu_isa_t isa, DataTypeSize S>
std::shared_ptr<jitGatherKernelInterface> createJitUniGatherKernel(uint64_t afterAxisSize, uint64_t specIdxSize, uint64_t idxElPerVec) {
    if (afterAxisSize == 1lu)
        return createJitUniGatherKernel<isa, S, ElementwiseCase>(specIdxSize < idxElPerVec);
    return createJitUniGatherKernel<isa, S, BlockedCase>(afterAxisSize <= idxElPerVec);
}

template <x64::cpu_isa_t isa, DataTypeSize S>
std::shared_ptr<jitGatherKernelInterface> createJitUniGatherKernel(
        uint64_t afterAxisSize, uint64_t specIdxSize, uint64_t idxElPerVec, bool isDynamicNode) {
    if (isDynamicNode) {
        if (afterAxisSize != 1lu) {
            return {}; // BlockedCase for dynamic shapes not implemented
        }
        return std::make_shared<jitGatherKernelForDynamicShapes<isa, S>>();
    }
    return createJitUniGatherKernel<isa, S>(afterAxisSize, specIdxSize, idxElPerVec);
}

template <x64::cpu_isa_t isa>
std::shared_ptr<jitGatherKernelInterface> createJitUniGatherKernel(
        uint64_t dataTypeSize, bool isDynamicNode, uint64_t afterAxisSize, uint64_t specIdxSize, uint64_t idxElPerVec) {
    if (dataTypeSize == 4) {
        return createJitUniGatherKernel<isa, DataType32bit>(afterAxisSize, specIdxSize, idxElPerVec, isDynamicNode);
    } else if (isa == x64::avx512_core || afterAxisSize == 1) {
        if (dataTypeSize == 2) {
            return createJitUniGatherKernel<isa, DataType16bit>(afterAxisSize, specIdxSize, idxElPerVec, isDynamicNode);
        }
        return createJitUniGatherKernel<isa, DataType8bit>(afterAxisSize, specIdxSize, idxElPerVec, isDynamicNode);
    }
    return {}; // not implemented case
}

std::shared_ptr<jitGatherKernelInterface> jitGatherKernelInterface::createJitUniGatherKernel(x64::cpu_isa_t isa,
        uint64_t dataTypeSize, bool isDynamicNode, uint64_t afterAxisSize, uint64_t specIdxSize, uint64_t idxElPerVec) {
    if (isa == x64::avx512_core)
        return intel_cpu::createJitUniGatherKernel<x64::avx512_core>(dataTypeSize, isDynamicNode, afterAxisSize, specIdxSize, idxElPerVec);
    return intel_cpu::createJitUniGatherKernel<x64::avx2>(dataTypeSize, isDynamicNode, afterAxisSize, specIdxSize, idxElPerVec);
}

}   // namespace intel_cpu
}   // namespace ov
