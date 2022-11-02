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
    simdVecSize = jcp.simdVecSize;
    idxElPerVec = jcp.idxElPerVec;
    dataElPerVec = jcp.dataElPerVec;
    beforeAxisSize = jcp.beforeAxisSize;
    specIdxSize = jcp.specIdxSize;
    batchDims = jcp.batchDims;
    reverseIndexing = jcp.reverseIndexing;
    afterAxisSize = jcp.afterAxisSize;
    dynamicShapes = jcp.dynamicShapes;
    isLessSimdRegistersCase = isa != x64::avx512_core && afterAxisSize != 1 && !dynamicShapes && getDataTypeSize() != 4;
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
    preamble();
    constexpr int vlen = x64::cpu_isa_traits<isa>::vlen;
    stackAllocator = std::make_shared<StackAllocator>(*this, vlen);
    mov(regSrc, ptr[regParams + GET_OFF(src)]);
    mov(regDst, ptr[regParams + GET_OFF(dst)]);
    mov(regIndices, ptr[regParams + GET_OFF(indices)]);
    mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);
    vmmZeros = RegisterValue<Vmm>{[&](Vmm& value){ uni_vpxor(value, value, value); }};
    vmmZeros.initialize(regPool, vmmZerosIdx);
    vmmAxisDim = RegisterValue<Vmm>{[&](Vmm& value){ uploadParamPtrWithVpbroadcastd(value, GET_OFF(axisDim)); }};
    vmmAxisDim.initialize(regPool, vmmAxisDimIdx);
    vmmSrcBeforeAxisSumB = RegisterValue<Vmm>{[&](Vmm& value){ }};
    vmmSrcBeforeAxisSumB.initialize(this->regPool, 8);

    if (dynamicShapes) {
        generateForDynamicShapes();
    } else {
        generateForStaticShapes();
    }
    stackAllocator->release();
    stackAllocator.reset();
    postamble();
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
    cmp(regWorkAmount, dataElPerVec);
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

        {
            RegistersPool::Reg<Vmm> vSrcShift;
            RegistersPool::Reg<Vmask> kGatherMask;
            std::tie(kGatherMask, vSrcShift) = shiftCalculator.calcSrcShift(*this, p > 0 || shiftFirst);
            auto kDstMask = fillRestWorkMask(regWorkAmount);
            combineMasks(kGatherMask, kDstMask);
            RegistersPool::Reg<Vmm> vmmData{regPool};
            uni_vmovups(vmmData, vmmZeros);
            uniVpGatherDd(vmmData, ptr[regSrc + vSrcShift], kGatherMask);
            if (getDataTypeSize() == 4) {
                uni_vmovups_tail(ptr[regDst], kDstMask, vmmData);
                sub(regWorkAmount, dataElPerVec);
            } else {
                vSrcShift.release();
                storeVectorPart(regDst, regWorkAmount, vmmData);
            }
        }
        if (isLessSimdRegistersCase) {
            vmmSrcBeforeAxisSumB.loadFromStack(regPool, 8);
            auto& vmmAxisAndAfterAxisSizeB =
                    dynamic_cast<typename jitGatherKernelBase<isa>::template ShiftCalculatorImpl<BlockedCase, Short>&>(shiftCalculator).
                            vmmAxisAndAfterAxisSizeB;
            if (vmmAxisAndAfterAxisSizeB.isInitialized()) {
                vmmAxisAndAfterAxisSizeB.loadFromStack(this->regPool, this->vmmAxisAndAfterAxisSizeBIndx);
            }
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
    if (isLessSimdRegistersCase) {
        vmmAxisDim.reset();
    }
    RegistersPool::Reg<Vmask> negativeIndexesMask{regPool};
    vpcmpgtd(negativeIndexesMask, vmmZeros, vRawIndices);
    vpandn(kDstMask, negativeIndexesMask, kDstMask);
    negativeIndexesMask.release();
    if (isLessSimdRegistersCase) {
        vmmAxisDim.initialize(regPool, vmmAxisDimIdx);
    }
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
poolVmask<x64::avx512_core> jitGatherKernelBase<x64::avx512_core>::fillRestWorkMask(const Xbyak::Reg& rWorkRest) {
    Xbyak::Label lKmov;
    RegistersPool::Reg<Xbyak::Reg32> rOnes {regPool};
    mov(rOnes, 0x0000FFFF);
    cmp(rWorkRest, idxElPerVec);
    jge(lKmov);
    Xbyak::Reg8 rShift(Xbyak::Operand::CL);
    mov(rShift, idxElPerVec);
    sub(rShift, rWorkRest);
    shr(rOnes, rShift);
    L(lKmov);
    RegistersPool::Reg<Vmask> kDstMask {regPool};
    kmovw(kDstMask, rOnes);
    return kDstMask;
}

template <>
poolVmask<x64::avx2> jitGatherKernelBase<x64::avx2>::fillRestWorkMask(const Xbyak::Reg& rWorkRest) {
    Xbyak::Label lEnd;
    RegistersPool::Reg<Xbyak::Reg64> rAux0 {regPool};
    mov(rAux0, rWorkRest);
    RegistersPool::Reg<Xbyak::Reg32> rOnes {regPool};
    mov(rOnes, 0xFFFFFFFF);
    RegistersPool::Reg<Xbyak::Xmm> xmmAux{regPool};
    RegistersPool::Reg<Vmask> kDstMask {regPool};
    uni_vmovups(kDstMask, vmmZeros);
    for (uint8_t i = 0; i < idxElPerVec; i++) {
        cmp(rAux0, 0);
        je(lEnd, T_NEAR);

        if (i % 4 == 0)
            uni_vmovups(xmmAux, Xbyak::Xmm(vmmZeros.getIdx()));

        vpinsrd(xmmAux, xmmAux, rOnes, i % 4);
        vinserti128(kDstMask, kDstMask, xmmAux, i / 4);
        sub(rAux0, 1);
    }
    L(lEnd);
    return kDstMask;
}

template <x64::cpu_isa_t isa>
void jitGatherKernelBase<isa>::storeVectorPart(const Xbyak::Reg& rDst, const Xbyak::Reg& rToStoreCounter, Vmm& vmmData) {
    static const uint32_t vlenXmm = x64::cpu_isa_traits<x64::sse41>::vlen;
    Xbyak::Label lEnd;
    RegistersPool::Reg<Xbyak::Xmm> xAux {regPool};
    for (int j = 0; j < simdVecSize / vlenXmm; j++) {
        if (isa == x64::avx2)
            vextracti128(xAux, vmmData, j);
        else if (isa == x64::avx512_core)
            vextracti64x2(xAux, vmmData, j);

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
        this->add(this->regDst, this->simdVecSize);
        this->sub(this->regWorkAmount, this->dataElPerVec);
        this->cmp(this->regWorkAmount, this->dataElPerVec);
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
        this->add(this->regDst, this->simdVecSize);
        this->sub(this->regWorkAmount, this->dataElPerVec);
        this->cmp(this->regWorkAmount, this->dataElPerVec);
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
    if (this->isLessSimdRegistersCase) {
        this->vmmSrcBeforeAxisSumB.loadFromStack(this->regPool, 8);
        auto& vmmAxisAndAfterAxisSizeB =
                dynamic_cast<typename jitGatherKernelBase<isa>::template ShiftCalculatorImpl<BlockedCase, Short>&>(shiftCalculator).
                vmmAxisAndAfterAxisSizeB;
        if (vmmAxisAndAfterAxisSizeB.isInitialized()) {
            vmmAxisAndAfterAxisSizeB.loadFromStack(this->regPool, this->vmmAxisAndAfterAxisSizeBIndx);
        }
    }
    {
        auto gatheredData = this->shiftIdxAndGather(shiftCalculator, true);
        RegistersPool::Reg<Vmm> vAux0{this->regPool};
        this->vpshufb(vAux0, gatheredData, vShufMask);
        this->vshufps(halfPart, halfPart, vAux0, 0x0);
    }
    if (this->isLessSimdRegistersCase) {
        this->vmmSrcBeforeAxisSumB.loadFromStack(this->regPool, 8);
        auto& vmmAxisAndAfterAxisSizeB =
                dynamic_cast<typename jitGatherKernelBase<isa>::template ShiftCalculatorImpl<BlockedCase, Short>&>(shiftCalculator).
                        vmmAxisAndAfterAxisSizeB;
        if (vmmAxisAndAfterAxisSizeB.isInitialized()) {
            vmmAxisAndAfterAxisSizeB.loadFromStack(this->regPool, this->vmmAxisAndAfterAxisSizeBIndx);
        }
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
        RegisterValue<Vmm> vPermMask{[&](Vmm& value){
                                         RegistersPool::Reg<Xbyak::Reg64> regAux1{this->regPool};
                                         if (isa == x64::avx2) {
                                             this->mov(regAux1, reinterpret_cast<uintptr_t>(permMask8bitA2));
                                         } else if (isa == x64::avx512_core) {
                                             this->mov(regAux1, reinterpret_cast<uintptr_t>(permMask8bitA5));
                                         }
                                         this->uni_vmovups(value, this->ptr[regAux1]);
                                     }};
        vPermMask.initialize(this->regPool);
        if (this->isLessSimdRegistersCase) {
            this->vmmZeros.reset();
        }
        {
            RegistersPool::Reg<Vmm> vAux0{this->regPool};
            this->vshufps(vAux0, vBuff0, vBuff1, 0x88);
            this->vpermd(vAux0, vPermMask, vAux0);
            if (isa != x64::avx512_core) {
                vPermMask.reset();
            }

            this->uni_vmovups(this->ptr[this->regDst], vAux0);
        }
        if (this->isLessSimdRegistersCase) {
            this->vmmZeros.initialize(this->regPool, this->vmmZerosIdx);
        }

        // Main loop.
        this->L(lDstIdxLoop1);
        {
            this->add(this->regDst, this->simdVecSize);
            this->sub(this->regWorkAmount, this->dataElPerVec);
            this->cmp(this->regWorkAmount, this->dataElPerVec);
            this->jl(lTail, this->T_NEAR);

            vBuff0 = calculateIdxShiftsForHalfIteration(shiftCalculator, vShufMask, true, std::move(vBuff0));
            vBuff1 = calculateIdxShiftsForHalfIteration(shiftCalculator, vShufMask, true, std::move(vBuff1));
            if (this->isLessSimdRegistersCase) {
                this->vmmZeros.reset();
            }
            {
                RegistersPool::Reg<Vmm> vAux0{this->regPool};
                this->vshufps(vAux0, vBuff0, vBuff1, 0x88);
                if (isa != x64::avx512_core) {
                    vPermMask.initialize(this->regPool);
                }
                this->vpermd(vAux0, vPermMask, vAux0);
                if (isa != x64::avx512_core) {
                    vPermMask.reset();
                }

                this->uni_vmovups(this->ptr[this->regDst], vAux0);
            }
            if (this->isLessSimdRegistersCase) {
                this->vmmZeros.initialize(this->regPool, this->vmmZerosIdx);
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
    static const unsigned incVec[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    Xbyak::Label lBlockedCase, lEnd;
    const Xbyak::Reg64& regAfterAxSize = this->rsi;
    this->mov(regAfterAxSize, this->ptr[this->regParams + GET_OFF(afterAxSize)]);
    this->cmp(regAfterAxSize, 1);
    this->jg(lBlockedCase, this->T_NEAR);
    {
        typename jitGatherKernelBase<isa>::template ShiftCalculatorImpl<ElementwiseCase, Long> elementwiseLong;
        typename jitGatherKernelBase<isa>::template ShiftCalculatorImpl<ElementwiseCase, Short> elementwiseShort;
        elementwiseLong.allocateRegisters(*this);

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

        Xbyak::Label lLessThanVector1, lTail1, lTail2, lE1;

        this->cmp(regSpecIdxSizeB, this->simdVecSize);
        this->jl(lLessThanVector1, this->T_NEAR);
        {
            this->uni_vmovd(Xbyak::Reg32(elementwiseLong.regIdxIter.getIdx()), Xbyak::Xmm(this->vmmSpecIdxB.getIdx()));
            this->fillVlenVector(elementwiseLong.vmmVecLenB);

            this->isLessSimdRegistersCase = false;
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

            this->isLessSimdRegistersCase = false;
            this->process(elementwiseShort);
            elementwiseShort.releaseRegisters();
        }
        this->L(lE1);
        this->jmp(lEnd, this->T_NEAR);
    }
    this->L(lBlockedCase); {
        Xbyak::Label lShortCase, lTail3, lTail4, lShortCaseEnd;
        this->cmp(regAfterAxSize, this->idxElPerVec);
        this->jle(lShortCase, this->T_NEAR); {
        }
        this->jmp(lShortCaseEnd, this->T_NEAR);
        this->L(lShortCase); {
            typename jitGatherKernelBase<isa>::template ShiftCalculatorImpl<BlockedCase, Short> blockedShort;
            this->uploadParamPtrWithVpbroadcastd(this->vmmSpecIdxSizeB, GET_OFF(specIndicesSize));
            this->uni_vpslld(this->vmmSpecIdxSizeB, this->vmmSpecIdxSizeB, this->idxTypeShift); // multiply by indexes type size.

            this->uploadParamPtrWithVmovups(this->vmmSpecIdxB, GET_OFF(specIdxB));

            if (this->beforeAxisSize != 1lu) {
                this->uploadParamPtrWithVmovups(this->vmmSrcBeforeAxisSumB, GET_OFF(dataBeforeAxisSumB));
            }
            this->isLessSimdRegistersCase = isa != x64::avx512_core && this->getDataTypeSize() != 4;
            blockedShort.allocateRegisters(*this);
            blockedShort.uploadParamsForApproachSpecific(*this);
            this->process(blockedShort);
        }
        this->L(lShortCaseEnd);
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
    mov(reg32Aux1, this->simdVecSize);
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
    vmmAxisAndAfterAxisSizeB = RegistersPool::Reg<Vmm>{kernel.regPool, kernel.vmmAxisAndAfterAxisSizeBIndx};
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
    if (kernel.dynamicShapes) {
        rSpecIdxAndAfterAxSize = RegistersPool::Reg<Xbyak::Reg64>{kernel.regPool, 14};
        rBeforeAxisSize = RegistersPool::Reg<Xbyak::Reg64>{kernel.regPool, 15};
        rSpecIdxAndAfterAxisSizeIsPowerOf2 = RegistersPool::Reg<Xbyak::Reg64>{kernel.regPool};
        rAfterAxisSizeIsPowerOf2 = RegistersPool::Reg<Xbyak::Reg64>{kernel.regPool};
    }

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

    if (kernel.dynamicShapes) {
        kernel.mov(rSpecIdxSize, kernel.ptr[kernel.regParams + GET_OFF(specIdxSize)]);
        kernel.mov(rAfterAxisSizeIsPowerOf2, kernel.ptr[kernel.regParams + GET_OFF(afterAxisSizeIsPowerOf2)]);
        kernel.mov(rSpecIdxAndAfterAxSize, kernel.ptr[kernel.regParams + GET_OFF(specIdxAndAfterAxSize)]);
        vmmAxisAndAfterAxisSizeB = RegisterValue<Vmm>{[&](Vmm &value) {
            Xbyak::Label lBeforeAxisDiff, lEnd;
            kernel.cmp(rSpecIdxAndAfterAxSize, kernel.idxElPerVec);
            kernel.jl(lBeforeAxisDiff, kernel.T_NEAR); {
                kernel.uploadParamPtrWithVpbroadcastd(value, GET_OFF(axisAndAfterAxisSizeB));
            }
            kernel.jmp(lEnd, kernel.T_NEAR);
            kernel.L(lBeforeAxisDiff); {
                kernel.uploadParamPtrWithVmovups(value, GET_OFF(beforeAxisDiff));
            }
            kernel.L(lEnd);
        }};
        kernel.mov(rBeforeAxisSize, kernel.ptr[kernel.regParams + GET_OFF(beforeAxisSize)]);
        kernel.mov(rSpecIdxAndAfterAxisSizeIsPowerOf2, kernel.ptr[kernel.regParams + GET_OFF(specIdxAndAfterAxisSizeIsPowerOf2)]);

        Xbyak::Label lEnd, lBeforeAxPermMaskUploadEnd;
        kernel.cmp(rBeforeAxisSize, 1);
        kernel.je(lEnd, kernel.T_NEAR); {
            kernel.mov(rSpecIdxAndAfterAxIterB, kernel.ptr[kernel.regParams + GET_OFF(specIdxAndAfterAxIterB)]);
            kernel.mov(rSpecIdxAndAfterAxSizeB, kernel.ptr[kernel.regParams + GET_OFF(specIdxAndAfterAxSizeB)]);
            vmmAxisAndAfterAxisSizeB.initialize(kernel.regPool, kernel.vmmAxisAndAfterAxisSizeBIndx);
            kernel.cmp(rSpecIdxAndAfterAxisSizeIsPowerOf2, 1);
            kernel.je(lBeforeAxPermMaskUploadEnd, kernel.T_NEAR); {
                kernel.uploadParamPtrWithVmovups(vmmBeforeAxPermMask, GET_OFF(beforeAxisPermMask));
            }
            kernel.L(lBeforeAxPermMaskUploadEnd);
        }
        kernel.L(lEnd);
    } else {
        vmmAxisAndAfterAxisSizeB = RegisterValue<Vmm>{[&](Vmm &value) {
            if (kernel.specIdxSize * kernel.afterAxisSize < kernel.idxElPerVec) {
                kernel.uploadParamPtrWithVmovups(value, GET_OFF(beforeAxisDiff));
            } else {
                kernel.uploadParamPtrWithVpbroadcastd(value, GET_OFF(axisAndAfterAxisSizeB));
            }
        }};
        if (kernel.beforeAxisSize != 1lu) {
            kernel.mov(rSpecIdxAndAfterAxIterB, kernel.ptr[kernel.regParams + GET_OFF(specIdxAndAfterAxIterB)]);
            kernel.mov(rSpecIdxAndAfterAxSizeB, kernel.ptr[kernel.regParams + GET_OFF(specIdxAndAfterAxSizeB)]);
            vmmAxisAndAfterAxisSizeB.initialize(kernel.regPool, kernel.vmmAxisAndAfterAxisSizeBIndx);
            const uint64_t specIdxAndAfterAxisSize = kernel.specIdxSize * kernel.afterAxisSize;
            if (specIdxAndAfterAxisSize != 1 && specIdxAndAfterAxisSize != 2 && specIdxAndAfterAxisSize != 4 &&
                specIdxAndAfterAxisSize != 8 && specIdxAndAfterAxisSize != 16) {
                kernel.uploadParamPtrWithVmovups(vmmBeforeAxPermMask, GET_OFF(beforeAxisPermMask));
            }
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
        if (isLessSimdRegistersCase) {
            vmmZeros.reset();
        }
        uniVpGatherDd(vDstShifts, ptr[regIndices + vAux0], calcAllOnesMask(vAux0));
    } else {
        vDstShifts = RegistersPool::Reg<Vmm> {regPool};
        if (isLessSimdRegistersCase) {
            vmmZeros.reset();
        }
        uniVpGatherDd(vDstShifts, ptr[regIndices + vmmSpecIdxB], calcAllOnesMask(vDstShifts));
    }
    if (isLessSimdRegistersCase) {
        vmmZeros.initialize(regPool, vmmZerosIdx);
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
void jitGatherKernelBase<x64::avx2>::normWithUpperBound(Vmm& vTarget, Vmm& vMax) {
    RegistersPool::Reg<Vmask> kAuxMask{regPool};
    vpcmpgtd(kAuxMask, vMax, vTarget);
    vpandn(kAuxMask, kAuxMask, vMax);
    uni_vpsubd(vTarget, vTarget, kAuxMask);
}

template <>
void jitGatherKernelBase<x64::avx512_core>::normWithUpperBound(Vmm& vTarget, Vmm& vMax) {
    RegistersPool::Reg<Vmask> kAuxMask{regPool};
    vpcmpd(kAuxMask, vMax, vTarget, 2); // 2 -> LE
    uni_vpsubd(vTarget | kAuxMask, vTarget, vMax);
}

template<x64::cpu_isa_t isa> template<typename unused>
std::tuple<poolVmask<isa> /*kDstMask*/, poolVmm<isa> /*vDstShifts*/>
jitGatherKernelBase<isa>::ShiftCalculatorImpl<BlockedCase, Short, unused>::calcSrcShift(
        jitGatherKernelBase& kernel, bool shiftFirst) {
    if (kernel.dynamicShapes) {
        RegistersPool::Reg<Vmm> vBeforeAxisSumB{kernel.regPool};
        if (shiftFirst) {
            if (kernel.isLessSimdRegistersCase) {
                kernel.vmmZeros.reset();
            }
            Xbyak::Label lNormSpecIdxBEnd;
            kernel.cmp(rSpecIdxSize, 1);
            kernel.je(lNormSpecIdxBEnd, kernel.T_NEAR); {
                kernel.uni_vpaddd(kernel.vmmSpecIdxB, kernel.vmmSpecIdxB, vmmSpecIdxDiff);
                kernel.normWithUpperBound(kernel.vmmSpecIdxB, kernel.vmmSpecIdxSizeB);
            }
            kernel.L(lNormSpecIdxBEnd);

            Xbyak::Label lPermAfterAxisIdxBEnd;
            kernel.cmp(rAfterAxisSizeIsPowerOf2, 1);
            kernel.je(lPermAfterAxisIdxBEnd, kernel.T_NEAR); {
                // No sense to permute if afterAxisSize is one of {1, 2, 4, 8, 16}
                kernel.vpermd(vmmAfterAxisIdxB, vmmAfterAxisPermMask, vmmAfterAxisIdxB);

                Xbyak::Label lPermSpecIdxDiffEnd;
                kernel.cmp(rSpecIdxSize, 1);
                kernel.je(lPermSpecIdxDiffEnd, kernel.T_NEAR); {
                    kernel.vpermd(vmmSpecIdxDiff, vmmAfterAxisPermMask, vmmSpecIdxDiff);
                }
                kernel.L(lPermSpecIdxDiffEnd);
            }
            kernel.L(lPermAfterAxisIdxBEnd);

            Xbyak::Label lBeforeAxisSumBCalculationEnd;
            kernel.cmp(rBeforeAxisSize, 1);
            kernel.je(lBeforeAxisSumBCalculationEnd, kernel.T_NEAR); {
                Xbyak::Label lGreaterThenVec, lGreaterThenVecEnd;
                kernel.cmp(rSpecIdxAndAfterAxSize, kernel.idxElPerVec);
                kernel.jg(lGreaterThenVec, kernel.T_NEAR); {
                    auto &vmmBeforeAxDiffB = vmmAxisAndAfterAxisSizeB;
                    kernel.uni_vpaddd(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB, vmmBeforeAxDiffB);
                    kernel.uni_vmovups(vBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB);

                    Xbyak::Label lBeforeAxDiffBPermEnd;
                    kernel.cmp(rSpecIdxAndAfterAxisSizeIsPowerOf2, 1);
                    kernel.je(lBeforeAxDiffBPermEnd, kernel.T_NEAR); {
                        kernel.vpermd(vmmBeforeAxDiffB, vmmBeforeAxPermMask, vmmBeforeAxDiffB);
                    }
                    kernel.L(lBeforeAxDiffBPermEnd);
                }
                kernel.jmp(lGreaterThenVecEnd, kernel.T_NEAR);
                kernel.L(lGreaterThenVec); {
                    Xbyak::Label lBeforeAxStep, lBeforeAxStepEnd;
                    kernel.add(rSpecIdxAndAfterAxIterB, kernel.idxElPerVec * kernel.getDataTypeSize());
                    kernel.cmp(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);
                    kernel.jl(lBeforeAxStep, kernel.T_NEAR); {
                        kernel.sub(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);

                        RegistersPool::Reg<Vmm> vAux0{kernel.regPool};
                        kernel.vpmulld(vAux0, kernel.vmmSpecIdxB, vmmAfterAxisSize);
                        kernel.uni_vpaddd(vAux0, vAux0, vmmAfterAxisIdxB);
                        kernel.uni_vpbroadcastd(vBeforeAxisSumB, Xbyak::Xmm(vAux0.getIdx()));
                        if (isa == x64::avx512_core) {
                            RegistersPool::Reg<Xbyak::Opmask> kMask0{kernel.regPool};
                            kernel.vpcmpgtd(kMask0, vBeforeAxisSumB, vAux0);
                            kernel.uni_vmovups(vBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB);
                            kernel.uni_vpaddd(static_cast<Vmm &>(vBeforeAxisSumB) | kMask0, kernel.vmmSrcBeforeAxisSumB,
                                              vmmAxisAndAfterAxisSizeB);
                        } else {
                            kernel.vpcmpgtd(vBeforeAxisSumB, vBeforeAxisSumB, vAux0);
                            kernel.vpand(vBeforeAxisSumB, vBeforeAxisSumB, vmmAxisAndAfterAxisSizeB);
                            kernel.uni_vpaddd(vBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB, vBeforeAxisSumB);
                        }
                        kernel.uni_vpaddd(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB,
                                          vmmAxisAndAfterAxisSizeB);
                    }
                    kernel.jmp(lBeforeAxStepEnd);
                    kernel.L(lBeforeAxStep); {
                        kernel.uni_vmovups(vBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB);
                    }
                    kernel.L(lBeforeAxStepEnd);
                }
                kernel.L(lGreaterThenVecEnd);
            }
            kernel.L(lBeforeAxisSumBCalculationEnd);
            if (kernel.isLessSimdRegistersCase) {
                kernel.vmmZeros.initialize(kernel.regPool, kernel.vmmZerosIdx);
            }
        } else {
            Xbyak::Label lBeforeAxisSumBCalculationEnd;
            kernel.cmp(rBeforeAxisSize, 1);
            kernel.je(lBeforeAxisSumBCalculationEnd, kernel.T_NEAR); {
                kernel.uni_vmovups(vBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB);


                Xbyak::Label lLessOrEqualThenVec;
                kernel.cmp(rSpecIdxAndAfterAxSize, kernel.idxElPerVec);
                kernel.jle(lLessOrEqualThenVec, kernel.T_NEAR); {
                    // Broadcast the last element.
                    if (isa == x64::avx512_core) {
                        kernel.vshuff64x2(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB,
                                          kernel.vmmSrcBeforeAxisSumB, 0xFF);
                    } else {
                        kernel.vpermq(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB, 0xFF);
                    }
                    kernel.vpshufd(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB, 0xFF);

                    Xbyak::Label lBeforeAxStepEnd1;
                    kernel.add(rSpecIdxAndAfterAxIterB, kernel.idxElPerVec * kernel.getDataTypeSize());
                    kernel.cmp(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);
                    kernel.jl(lBeforeAxStepEnd1, kernel.T_NEAR); {
                        kernel.sub(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);
                        kernel.cmp(rSpecIdxAndAfterAxIterB, 0);
                        kernel.jne(lBeforeAxStepEnd1, kernel.T_NEAR); {
                            kernel.uni_vpaddd(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB,
                                              vmmAxisAndAfterAxisSizeB);
                        }
                    }
                    kernel.L(lBeforeAxStepEnd1);
                }
                kernel.L(lLessOrEqualThenVec);
            }
            kernel.L(lBeforeAxisSumBCalculationEnd);
        }
        if (kernel.isLessSimdRegistersCase) {
            kernel.vmmSrcBeforeAxisSumB.saveToStack(kernel.stackAllocator);
            if (vmmAxisAndAfterAxisSizeB.isInitialized()) {
                vmmAxisAndAfterAxisSizeB.saveToStack(kernel.stackAllocator);
            }
        }

        RegistersPool::Reg<Vmm> vDstShifts;
        RegistersPool::Reg<Vmask> kDstMask;
        std::tie(kDstMask, vDstShifts) = kernel.calculateIndexesForShortCase(vBeforeAxisSumB, vmmSrcAfterBatchSizeB);

        kernel.vpmulld(vDstShifts, vDstShifts, vmmAfterAxisSize);
        kernel.uni_vpaddd(vDstShifts, vDstShifts, vmmAfterAxisIdxB);

        Xbyak::Label lBeforeAxisSumBAdditionEnd;
        kernel.cmp(rBeforeAxisSize, 1);
        kernel.je(lBeforeAxisSumBAdditionEnd, kernel.T_NEAR); {
            kernel.uni_vpaddd(vDstShifts, vDstShifts, vBeforeAxisSumB);
        }
        kernel.L(lBeforeAxisSumBAdditionEnd);
        return {std::move(kDstMask), std::move(vDstShifts)};
    } else {
        RegistersPool::Reg<Vmm> vBeforeAxisSumB{kernel.regPool};
        const uint64_t specIdxAndAfterAxisSize = kernel.specIdxSize * kernel.afterAxisSize;
        if (shiftFirst) {
            if (kernel.isLessSimdRegistersCase) {
                kernel.vmmZeros.reset();
            }
            if (kernel.specIdxSize != 1) {
                kernel.uni_vpaddd(kernel.vmmSpecIdxB, kernel.vmmSpecIdxB, vmmSpecIdxDiff);
                kernel.normWithUpperBound(kernel.vmmSpecIdxB, kernel.vmmSpecIdxSizeB);
            }
            // No sense to permute if afterAxisSize is one of {1, 2, 4, 8, 16}. 0 is reserved for dynamic case.
            if (kernel.afterAxisSize != 2 && kernel.afterAxisSize != 4 &&
                kernel.afterAxisSize != 8 && kernel.afterAxisSize != 16) {
                kernel.vpermd(vmmAfterAxisIdxB, vmmAfterAxisPermMask, vmmAfterAxisIdxB);
                if (kernel.specIdxSize != 1)
                    kernel.vpermd(vmmSpecIdxDiff, vmmAfterAxisPermMask, vmmSpecIdxDiff);
            }

            if (kernel.beforeAxisSize != 1lu) {
                if (specIdxAndAfterAxisSize > 0lu && specIdxAndAfterAxisSize <= kernel.idxElPerVec) {
                    auto &vmmBeforeAxDiffB = vmmAxisAndAfterAxisSizeB;
                    kernel.uni_vpaddd(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB, vmmBeforeAxDiffB);
                    kernel.uni_vmovups(vBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB);
                    if (specIdxAndAfterAxisSize != 1 && specIdxAndAfterAxisSize != 2 &&
                        specIdxAndAfterAxisSize != 4 &&
                        specIdxAndAfterAxisSize != 8 && specIdxAndAfterAxisSize != 16)
                        kernel.vpermd(vmmBeforeAxDiffB, vmmBeforeAxPermMask, vmmBeforeAxDiffB);
                } else {
                    Xbyak::Label lBeforeAxStep, lBeforeAxStepEnd;
                    kernel.add(rSpecIdxAndAfterAxIterB, kernel.idxElPerVec * kernel.getDataTypeSize());
                    kernel.cmp(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);
                    kernel.jl(lBeforeAxStep, kernel.T_NEAR);
                    kernel.sub(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);

                    RegistersPool::Reg<Vmm> vAux0{kernel.regPool};
                    kernel.vpmulld(vAux0, kernel.vmmSpecIdxB, vmmAfterAxisSize);
                    kernel.uni_vpaddd(vAux0, vAux0, vmmAfterAxisIdxB);
                    kernel.uni_vpbroadcastd(vBeforeAxisSumB, Xbyak::Xmm(vAux0.getIdx()));
                    if (isa == x64::avx512_core) {
                        RegistersPool::Reg<Xbyak::Opmask> kMask0{kernel.regPool};
                        kernel.vpcmpgtd(kMask0, vBeforeAxisSumB, vAux0);
                        kernel.uni_vmovups(vBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB);
                        kernel.uni_vpaddd(static_cast<Vmm &>(vBeforeAxisSumB) | kMask0, kernel.vmmSrcBeforeAxisSumB,
                                          vmmAxisAndAfterAxisSizeB);
                    } else {
                        kernel.vpcmpgtd(vBeforeAxisSumB, vBeforeAxisSumB, vAux0);
                        kernel.vpand(vBeforeAxisSumB, vBeforeAxisSumB, vmmAxisAndAfterAxisSizeB);
                        kernel.uni_vpaddd(vBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB, vBeforeAxisSumB);
                    }
                    kernel.uni_vpaddd(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB,
                                      vmmAxisAndAfterAxisSizeB);
                    kernel.jmp(lBeforeAxStepEnd);
                    kernel.L(lBeforeAxStep);
                    kernel.uni_vmovups(vBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB);
                    kernel.L(lBeforeAxStepEnd);
                }
            }
            if (kernel.isLessSimdRegistersCase) {
                kernel.vmmZeros.initialize(kernel.regPool, kernel.vmmZerosIdx);
            }
        } else {
            if (kernel.beforeAxisSize != 1lu) {
                kernel.uni_vmovups(vBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB);
                if (specIdxAndAfterAxisSize > kernel.idxElPerVec) {
                    // Broadcast the last element.
                    if (isa == x64::avx512_core) {
                        kernel.vshuff64x2(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB,
                                          kernel.vmmSrcBeforeAxisSumB, 0xFF);
                    } else {
                        kernel.vpermq(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB, 0xFF);
                    }
                    kernel.vpshufd(kernel.vmmSrcBeforeAxisSumB, kernel.vmmSrcBeforeAxisSumB, 0xFF);

                    Xbyak::Label lBeforeAxStepEnd1;
                    kernel.add(rSpecIdxAndAfterAxIterB, kernel.idxElPerVec * kernel.getDataTypeSize());
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
        if (kernel.isLessSimdRegistersCase) {
            kernel.vmmSrcBeforeAxisSumB.saveToStack(kernel.stackAllocator);
            if (vmmAxisAndAfterAxisSizeB.isInitialized()) {
                vmmAxisAndAfterAxisSizeB.saveToStack(kernel.stackAllocator);
            }
        }

        RegistersPool::Reg<Vmm> vDstShifts;
        RegistersPool::Reg<Vmask> kDstMask;
        std::tie(kDstMask, vDstShifts) = kernel.calculateIndexesForShortCase(vBeforeAxisSumB, vmmSrcAfterBatchSizeB);

        kernel.vpmulld(vDstShifts, vDstShifts, vmmAfterAxisSize);
        kernel.uni_vpaddd(vDstShifts, vDstShifts, vmmAfterAxisIdxB);
        if (kernel.beforeAxisSize != 1lu)
            kernel.uni_vpaddd(vDstShifts, vDstShifts, vBeforeAxisSumB);
        return {std::move(kDstMask), std::move(vDstShifts)};
    }
}

template<x64::cpu_isa_t isa> template<typename unused>
std::tuple<poolVmask<isa> /*kDstMask*/, poolVmm<isa> /*vDstShifts*/>
jitGatherKernelBase<isa>::ShiftCalculatorImpl<ElementwiseCase, Long, unused>::calcSrcShift(
        jitGatherKernelBase& kernel, bool shiftFirst) {
    if (isa == x64::avx2) {
        Xbyak::Label lIdxStride, lExit;
        if (shiftFirst)
            kernel.uni_vpaddd(kernel.vmmSpecIdxB, kernel.vmmSpecIdxB, vmmVecLenB);
        kernel.add(regIdxIter, kernel.simdVecSize);
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
        kernel.add(regIdxIter, kernel.simdVecSize);
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
        if (afterAxisSize != 1lu && afterAxisSize > idxElPerVec) {
            return {}; // Blocked/Long case for dynamic shapes not implemented
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
    } else {
        if (dataTypeSize == 2) {
            return createJitUniGatherKernel<isa, DataType16bit>(afterAxisSize, specIdxSize, idxElPerVec, isDynamicNode);
        }
        return createJitUniGatherKernel<isa, DataType8bit>(afterAxisSize, specIdxSize, idxElPerVec, isDynamicNode);
    }
}

std::shared_ptr<jitGatherKernelInterface> jitGatherKernelInterface::createJitUniGatherKernel(x64::cpu_isa_t isa,
        uint64_t dataTypeSize, bool isDynamicNode, uint64_t afterAxisSize, uint64_t specIdxSize, uint64_t idxElPerVec) {
    if (isa == x64::avx512_core)
        return intel_cpu::createJitUniGatherKernel<x64::avx512_core>(dataTypeSize, isDynamicNode, afterAxisSize, specIdxSize, idxElPerVec);
    return intel_cpu::createJitUniGatherKernel<x64::avx2>(dataTypeSize, isDynamicNode, afterAxisSize, specIdxSize, idxElPerVec);
}

}   // namespace intel_cpu
}   // namespace ov
