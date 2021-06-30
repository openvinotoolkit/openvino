// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_uni_kernel.hpp"

using namespace MKLDNNPlugin;
using namespace mkldnn::impl::cpu;

#define GET_OFF(field) offsetof(gatherJitExecArgs, field)

template <x64::cpu_isa_t isa>
jitUniGatherKernel<isa>::jitUniGatherKernel(jGatherConfParams jcp) :
        jitGatherKernelBase(jcp), x64::jit_generator() {
    vlen = x64::cpu_isa_traits<isa>::vlen;
    elPerVec = vlen / jcp.dataTypeSize;
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::create_ker() {
    x64::jit_generator::create_kernel();
    ker_ = (decltype(ker_))jit_ker();
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::generate() {
    this->preamble();

    mov(regSrc, ptr[regParams + GET_OFF(src)]);
    mov(regDst, ptr[regParams + GET_OFF(dst)]);
    mov(regIndices, ptr[regParams + GET_OFF(indices)]);

    mov(regIdxIter, ptr[regParams + GET_OFF(idxIter)]);

//    mov(regIdxIter, ptr[regParams + GET_OFF(idxStartB)]);

    mov(regAux1, ptr[regParams + GET_OFF(dataTypeSize)]);
    uni_vpbroadcastd(vmmDictTypeSize, ptr[regAux1]);

    mov(regAux1, ptr[regParams + GET_OFF(idxTypeSize)]);
    uni_vpbroadcastd(vmmAux3, ptr[regAux1]);

//    mov(regAux1, ptr[regParams + GET_OFF(axisDim)]);
//    uni_vpbroadcastd(vmmAxDim, ptr[regAux1]);

//    mov(regAux1, ptr[regParams + GET_OFF(axDimSum)]);
//    uni_vpbroadcastd(vmmAxDimSum, ptr[regAux1]);

    mov(regAux1, ptr[regParams + GET_OFF(minusOne)]);
    uni_vpbroadcastd(vmmMinusOne, ptr[regAux1]);

    mov(regAux1, ptr[regParams + GET_OFF(vecLen)]);
    uni_vpbroadcastd(vmmVecLen, ptr[regAux1]);

    mov(regAux1, ptr[regParams + GET_OFF(specIndicesSizePtr)]);
    uni_vpbroadcastd(vmmSpecIdxSize, ptr[regAux1]);
//    uni_vpmulld(vmmSpecIdxSize, vmmSpecIdxSize, vmmAux3);

    mov(regAux1, ptr[regParams + GET_OFF(batchIndices)]);
    uni_vmovups(vmmIdxBatchSum, ptr[regAux1]);

    mov(regAux1, ptr[regParams + GET_OFF(betweenBatchAndAxisIdx)]);
    uni_vmovups(vmmBeforeAxisSum, ptr[regAux1]);
    mov(regAux1, ptr[regParams + GET_OFF(axisAndAfterAxisSize)]);
    uni_vpbroadcastd(vmmAxisAndAfterAxisSize, ptr[regAux1]);
    uni_vpmulld(vmmBeforeAxisSum, vmmBeforeAxisSum, vmmAxisAndAfterAxisSize);
    mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSize)]);
    uni_vpbroadcastd(vmmAux0, ptr[regAux1]);
    uni_vpmulld(vmmAux0, vmmAux0, vmmIdxBatchSum);
    uni_vpaddd(vmmBeforeAxisSum, vmmBeforeAxisSum, vmmAux0);
//    uni_vpmulld(vmmBeforeAxisSum, vmmBeforeAxisSum, vmmDictTypeSize);

    uni_vpmulld(vmmIdxBatchSum, vmmIdxBatchSum, vmmSpecIdxSize);

    mov(regAux1, ptr[regParams + GET_OFF(specIndices)]);
    uni_vmovups(vmmSpecIndices, ptr[regAux1]);
    uni_vpmulld(vmmSpecIndices, vmmSpecIndices, vmmAux3);

    mov(regBeforeAxisCounter, ptr[regParams + GET_OFF(beforeAxisCounter)]);
    mov(regBetweenBatchAndAxisSize, ptr[regParams + GET_OFF(betweenBatchAndAxisSize)]);

    mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);

//        mov(regAux1, ptr[regParams + GET_OFF(tmp)]);
//        mov(regAux2, ptr[regParams + GET_OFF(retVal)]);

    elPerVec = vlen / jcp_.dataTypeSize;
    if (isa == x64::avx512_common) {
        vpcmpub(kMaskOnes, vmmGatherMask, vmmGatherMask, 0);
    }

//    uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);

    mov(regAux1, ptr[regParams + GET_OFF(afterAxisBlockSize)]);
    Xbyak::Label lBlock_N;
    cmp(regAux1, 1);
    jg(lBlock_N, T_NEAR);
    {
        if (jcp_.dataTypeSize == 4) {
            Xbyak::Label lTail;
            cmp(regWorkAmount, elPerVec);
            jl(lTail, T_NEAR);

            Xbyak::Label lLessThanVector;
            mov(regAux1, ptr[regParams + GET_OFF(specIndicesSize)]);
            cmp(regAux1, vlen);
            jl(lLessThanVector, T_NEAR);
            Xbyak::Label lDstIdxLoop1;

            // First iteration
            uni_vpaddd(vmmAux0, vmmIdxBatchSum, vmmSpecIndices);
            uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
            vpgatherdd(vmmAux1, ptr[regIndices + vmmAux0], vmmOnes);
//            uni_vmovups(ptr[regDst], vmmAux1);

            // Check boundaries
    //        vpcmpgtd(mask, dstIndices, vmmMinusOne);
    //        vpcmpgtd(vmmAux1, vmmAxDim, dstIndices);
    //        vpand(mask, mask, vmmAux1);

            uni_vpmulld(vmmAux1, vmmAux1, vmmDictTypeSize);
            uni_vpaddd(vmmAux0, vmmAux1, vmmBeforeAxisSum);
            uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
            vpgatherdd(vmmDst, ptr[regSrc + vmmAux0], vmmOnes);
//            uni_vmovups(ptr[regDst], vmmIdxBatchSum);
            uni_vmovups(ptr[regDst], vmmDst);

            add(regIdxIter, vlen);
            add(regDst, vlen);
            sub(regWorkAmount, elPerVec);

//        uni_vpaddd(vmmSpecIndices, vmmSpecIndices, vmmVecLen);
//        uni_vpaddd(vmmAux0, vmmIdxBatchSum, vmmSpecIndices); // combine with previous line?
//        uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
//        vpgatherdd(vmmAux1, ptr[regIndices + vmmAux0], vmmOnes);
//        uni_vmovups(ptr[regDst], vmmAux1);

            L(lDstIdxLoop1);
            {
                cmp(regWorkAmount, elPerVec);
                jl(lTail, T_NEAR);

                vpGatherDDlong(vmmDst);
                uni_vmovups(ptr[regDst], vmmDst);

                add(regDst, vlen);
                sub(regWorkAmount, elPerVec);

                jmp(lDstIdxLoop1, T_NEAR);
            }
            L(lLessThanVector);
            Xbyak::Label lDstIdxLoop;
            L(lDstIdxLoop);
            {
//                cmp(regWorkAmount, elPerVec);
//                jl(lTail, T_NEAR);
//
//                vpGatherDD(vmmDst);
//                uni_vmovups(ptr[regDst], vmmDst);
//
//                add(regDst, vlen);
//                sub(regWorkAmount, elPerVec);
//
//                jmp(lDstIdxLoop, T_NEAR);
            }
            L(lTail);
//            tail();
        } else if (jcp_.dataTypeSize == 2) {
            auto& vmmShufMask = vmmAux8;
            mov(regAux1, ptr[regParams + GET_OFF(shufMask16bitUni)]);
            uni_vmovups(vmmShufMask, ptr[regAux1]);

            auto& vmmPermMask = vmmAux9;
            if (isa == x64::avx512_common) {
                mov(regAux1, ptr[regParams + GET_OFF(permMask16bitA5)]);
            } else {
                mov(regAux1, ptr[regParams + GET_OFF(permMask16bitA2)]);
            }
            uni_vmovups(vmmPermMask, ptr[regAux1]);

            Xbyak::Label lDstIdxLoop, lTail;
            L(lDstIdxLoop);
            {
                cmp(regWorkAmount, elPerVec);
                jl(lTail, T_NEAR);

                // TODO: On AVX512_VBMI can be replaced on VPERMB(VPERMB(Gather()), Gather())
                gatherAndGroup(vmmDst, vmmShufMask);
                gatherAndGroup(vmmAux4, vmmShufMask);
                vshufps(vmmDst, vmmDst, vmmAux4, 0x44);
                vpermd(vmmDst, vmmPermMask, vmmDst);

                uni_vmovups(ptr[regDst], vmmDst);

                add(regDst, vlen);
                sub(regWorkAmount, elPerVec);

                jmp(lDstIdxLoop, T_NEAR);
            }
            L(lTail);
            tail();
        } else if (jcp_.dataTypeSize == 1) {
            auto& vmmShufMask = vmmAux8;
            mov(regAux1, ptr[regParams + GET_OFF(shufMask8bitUni)]);
            uni_vmovups(vmmShufMask, ptr[regAux1]);

            auto& vmmPermMask = vmmAux9;
            if (isa == x64::avx512_common) {
                mov(regAux1, ptr[regParams + GET_OFF(permMask8bitA5)]);
            } else {
                mov(regAux1, ptr[regParams + GET_OFF(permMask8bitA2)]);
            }
            uni_vmovups(vmmPermMask, ptr[regAux1]);

            Xbyak::Label lDstIdxLoop, lTail;
            L(lDstIdxLoop);
            {
                cmp(regWorkAmount, elPerVec);
                jl(lTail, T_NEAR);

                gatherAndGroup(vmmDst, vmmShufMask);
                gatherAndGroup(vmmAux4, vmmShufMask);
                vshufps(vmmDst, vmmDst, vmmAux4, 0);

                gatherAndGroup(vmmAux4, vmmShufMask);
                gatherAndGroup(vmmAux5, vmmShufMask);
                vshufps(vmmAux4, vmmAux4, vmmAux5, 0);

                vshufps(vmmDst, vmmDst, vmmAux4, 0x88);
                vpermd(vmmDst, vmmPermMask, vmmDst);

                uni_vmovups(ptr[regDst], vmmDst);

                add(regDst, vlen);
                sub(regWorkAmount, elPerVec);

                jmp(lDstIdxLoop, T_NEAR);
            }
            L(lTail);
            tail();
        }
    }
    L(lBlock_N);
    // else if (jcp_.afterAxisSize >= elPerVec) {
//            Xbyak::Label lDstIdxLoop, lTail;
//            L(lDstIdxLoop);
//            {
//                cmp(regWorkAmount, elPerVec);
//                jl(lTail, T_NEAR);
//
//                vpGatherDD(vmmDst);
//                uni_vmovups(ptr[regDst], vmmDst);
//
//                add(regDst, vlen);
//                sub(regWorkAmount, elPerVec);
//
//                jmp(lDstIdxLoop, T_NEAR);
//            }
//            L(lTail);
//            tail();
//    } else {
    {
//        Xbyak::Label lDstIdxLoop, lTail;
//        L(lDstIdxLoop);
//        {
//            cmp(regWorkAmount, elPerVec);
//            jl(lTail, T_NEAR);
//
//            vpGatherDDBlk(vmmDst);
//            uni_vmovups(ptr[regDst], vmmDst);
//
//            add(regDst, vlen);
//            sub(regWorkAmount, elPerVec);
//
//            jmp(lDstIdxLoop, T_NEAR);
//        }
//        L(lTail);
//        tail();
    }

    this->postamble();
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::tail() {
    Xbyak::Label lTailLoop, lCalc, lCmpTop, lPositiveIdx, lFinish;
    Xbyak::Reg32 regDictTypeSize32(regAux1.getIdx());
    Xbyak::Reg32 regAxDimSum32(regAux2.getIdx());
    Xbyak::Reg32 regAux3_32(regAux3.getIdx());
    Xbyak::Reg16 regAux3_16(regAux3.getIdx());
    Xbyak::Reg8  regAux3_8(regAux3.getIdx());
    uni_vpextrd(regDictTypeSize32, xmmDictTypeSize, 0);
    uni_vpextrd(regAxDimSum32, xmmAxDimSum, 0);
    L(lTailLoop);
    {
        cmp(regWorkAmount, 0);
        je(lFinish, T_NEAR);

        cmp(regIdxIter, jcp_.indicesSize);
        jl(lCalc, T_NEAR);
        mov(regIdxIter, 0);
        uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
        uni_vpextrd(regAxDimSum32, xmmAxDimSum, 0);

        L(lCalc);
        mov(eax, ptr[regIndices + regIdxIter]);
        cmp(rax, 0);
        jge(lCmpTop, T_NEAR);
        mov(rax, 0);
        jmp(lPositiveIdx);
        L(lCmpTop);
        cmp(rax, jcp_.axisDim);
        jl(lPositiveIdx, T_NEAR);
        mov(rax, 0);
        L(lPositiveIdx);
        mul(regDictTypeSize32);
        add(eax, regAxDimSum32);
        if (jcp_.dataTypeSize == 4) {
            mov(regAux3_32, ptr[regSrc + rax]);
            mov(ptr[regDst], regAux3_32);
        } else if (jcp_.dataTypeSize == 2) {
            mov(regAux3_16, ptr[regSrc + rax]);
            mov(ptr[regDst], regAux3_16);
        } else if (jcp_.dataTypeSize == 1) {
            mov(regAux3_8, ptr[regSrc + rax]);
            mov(ptr[regDst], regAux3_8);
        }

        add(regIdxIter, sizeof(int));
        add(regDst, jcp_.dataTypeSize);
        sub(regWorkAmount, 1);
        jmp(lTailLoop, T_NEAR);
    }
    L(lFinish);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::fillIndicies(Xbyak::Xmm& dst, Xbyak::Xmm& mask) {
    Xbyak::Label lPerElements, lExit;

    mov(regAux1, ptr[regParams + GET_OFF(specIndicesSize)]);
    mov(regAux2, regAux1);
    sub(regAux2, vlenXmm);
    cmp(regIdxIter, regAux2);
    jg(lPerElements, T_NEAR);
        uni_vmovups(dst, ptr[regIndices + regIdxIter]);
        uni_vpmulld(dst, dst, xmmDictTypeSize);
        // Check boundaries
        vpcmpgtd(mask, dst, xmmMinusOne);
        vpcmpgtd(xmmAux1, xmmAxDim, dst);
        vpand(mask, mask, xmmAux1);

        uni_vpaddd(dst, dst, xmmAxDimSum);
        add(regIdxIter, vlenXmm);
    cmp(regIdxIter, regAux1);
    jl(lExit, T_NEAR);
        uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
        mov(regIdxIter, 0);
    jmp(lExit, T_NEAR);

    L(lPerElements);
    for (uint8_t i = 0; i < 4; i++) {
        Xbyak::Label insertLabel;

        cmp(regIdxIter, regAux1);
        jl(insertLabel, T_NEAR);
            mov(regIdxIter, 0);
            uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);

        L(insertLabel);
        uni_vpbroadcastd(xmmAux1, ptr[regIndices + regIdxIter]);
        uni_vpmulld(xmmAux1, xmmAux1, xmmDictTypeSize);
        vpcmpgtd(xmmAux3, xmmAux1, xmmMinusOne);
        vpcmpgtd(xmmAux7, xmmAxDim, xmmAux1);
        vpand(xmmAux3, xmmAux3, xmmAux7);
        uni_vpaddd(xmmAux1, xmmAux1, xmmAxDimSum);
        vinsertps(dst, dst, xmmAux1, i << 4);
        vinsertps(mask, mask, xmmAux3, i << 4);
        add(regIdxIter, sizeof(int));
    }
    L(lExit);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::fillIndicies(Xbyak::Ymm& dstIndices, Xbyak::Ymm& mask) {
    Xbyak::Label lPerXmm, lExit;

    mov(regAux1, ptr[regParams + GET_OFF(specIndicesSize)]);
    mov(regAux2, regAux1);
    sub(regAux2, vlenYmm);
    cmp(regIdxIter, regAux2);
    jg(lPerXmm, T_NEAR);
        uni_vmovups(dstIndices, ptr[regIndices + regIdxIter]);
        uni_vpmulld(dstIndices, dstIndices, vmmDictTypeSize);
        // Check boundaries
        vpcmpgtd(mask, dstIndices, vmmMinusOne);
        vpcmpgtd(vmmAux1, vmmAxDim, dstIndices);
        vpand(mask, mask, vmmAux1);

        uni_vpaddd(dstIndices, dstIndices, vmmAxDimSum);
        add(regIdxIter, vlenYmm);
    cmp(regIdxIter, regAux1);
    jl(lExit, T_NEAR);
        uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
        mov(regIdxIter, 0);
    jmp(lExit, T_NEAR);
    L(lPerXmm);
        for (int i = 0; i < 2; i++) {
            fillIndicies(xmmAux0, xmmAux2);
            vinsertf128(dstIndices, dstIndices, xmmAux0, i);
            vinsertf128(mask, mask, xmmAux2, i);
        }
    L(lExit);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::fillIndicies(Xbyak::Zmm& dst, Xbyak::Opmask& mask) {
    Xbyak::Label lPerYmm, lExit;

    cmp(regIdxIter, jcp_.indicesSize - vlen);
    jg(lPerYmm, T_NEAR);
        uni_vmovups(dst, ptr[regIndices + regIdxIter]);
        uni_vpmulld(dst, dst, vmmDictTypeSize);
    vpcmpgtd(mask, dst, vmmMinusOne);
    vpcmpgtd(kMaskAux2, vmmAxDim, dst);
    kandd(mask, mask, kMaskAux2);
        uni_vpaddd(dst, dst, vmmAxDimSum);
        add(regIdxIter, vlen);
    cmp(regIdxIter, jcp_.indicesSize);
    jl(lExit, T_NEAR);
        uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
        mov(regIdxIter, 0);
    jmp(lExit, T_NEAR);
    L(lPerYmm);
        for (int i = 0; i < 2; i++) {
            fillIndicies(ymmAux2, ymmAux10);
            vinsertf32x8(dst, dst, ymmAux2, i);
            vinsertf32x8(vmmAux11, vmmAux11, ymmAux10, i);
        }
        vpmovd2m(mask, ymmAux10);
    L(lExit);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::fillIndiciesLong(Xbyak::Ymm& dstIndices, Xbyak::Ymm& mask) {
    Xbyak::Label lPerXmm, lExit;
    uni_vpcmpeqd(mask, mask, mask);

    mov(regAux1, ptr[regParams + GET_OFF(specIndicesSize)]);
    add(regIdxIter, vlenYmm);
    cmp(regIdxIter, regAux1);
    jg(lPerXmm, T_NEAR);
        uni_vpaddd(vmmSpecIndices, vmmSpecIndices, vmmVecLen);
        uni_vpaddd(vmmAux0, vmmIdxBatchSum, vmmSpecIndices); // combine with previous line?
        uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
        vpgatherdd(vmmAux1, ptr[regIndices + vmmAux0], vmmOnes);
        // Check boundaries
//        vpcmpgtd(mask, dstIndices, vmmMinusOne);
//        vpcmpgtd(vmmAux1, vmmAxDim, dstIndices);
//        vpand(mask, mask, vmmAux1);
        uni_vpmulld(vmmAux1, vmmAux1, vmmDictTypeSize);
//uni_vmovups(dstIndices, vmmIdxBatchSum);

        uni_vpaddd(dstIndices, vmmBeforeAxisSum, vmmAux1);
//    cmp(regIdxIter, regAux1);
//    jl(lExit, T_NEAR);
//        uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
//        mov(regIdxIter, 0);
    jmp(lExit, T_NEAR);
    L(lPerXmm);
        sub(regIdxIter, regAux1);
        uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
        // increment indices
        uni_vpaddd(vmmSpecIndices, vmmSpecIndices, vmmVecLen);
//        vpcmpgtd(vmmAux0, vmmSpecIndices, vmmSpecIdxSize); // SpecIdxSize - 1
        vpcmpgtd(vmmAux0, vmmSpecIdxSize, vmmSpecIndices);
        uni_vpsubd(vmmAux0, vmmOnes, vmmAux0);
        uni_vpand(vmmAux1, vmmAux0, vmmSpecIdxSize);
        uni_vpsubd(dstIndices, vmmSpecIndices, vmmAux1);
        uni_vpsubd(vmmSpecIndices, vmmSpecIndices, vmmSpecIdxSize);

        Xbyak::Label l1, l2;
        inc(regBeforeAxisCounter);
        cmp(regBeforeAxisCounter, regBetweenBatchAndAxisSize);
        jl(l1, T_NEAR);
            mov(regBeforeAxisCounter, 0);
            uni_vpand(vmmAux1, vmmAux0, vmmSpecIdxSize);
            uni_vpaddd(vmmAux1, vmmIdxBatchSum, vmmAux1);
            uni_vpaddd(vmmIdxBatchSum, vmmIdxBatchSum, vmmSpecIdxSize);
            uni_vpaddd(vmmAux1, vmmAux1, dstIndices);
            jmp(l2, T_NEAR);
        L(l1);
            uni_vpaddd(vmmAux1, vmmIdxBatchSum, dstIndices);
        L(l2);

//        uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
//uni_vmovups(dstIndices, vmmIdxBatchSum);
        vpgatherdd(vmmAux3, ptr[regIndices + vmmAux1], vmmOnes);
        uni_vpmulld(vmmAux1, vmmAux3, vmmDictTypeSize);
        // TODO: check negative indices

        uni_vpand(vmmAux0, vmmAux0, vmmAxisAndAfterAxisSize);
        uni_vpaddd(vmmAux0, vmmAux0, vmmBeforeAxisSum);
        uni_vpaddd(vmmBeforeAxisSum, vmmBeforeAxisSum, vmmAxisAndAfterAxisSize);
//uni_vmovups(dstIndices, vmmBeforeAxisSum);

        uni_vpaddd(dstIndices, vmmAux0, vmmAux1);
    L(lExit);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::fillIndiciesBlk(Xbyak::Ymm& dst, Xbyak::Ymm& mask) {
    Xbyak::Label lPerXmm, lExit;

    cmp(regIdxIter, jcp_.indicesSize - vlenYmm);
    jg(lPerXmm, T_NEAR);
        uni_vmovups(dst, ptr[regIndices + regIdxIter]);
        uni_vpmulld(dst, dst, vmmDictTypeSize);
        // Check boundaries
        vpcmpgtd(mask, dst, vmmMinusOne);
        vpcmpgtd(vmmAux1, vmmAxDim, dst);
        vpand(mask, mask, vmmAux1);

        uni_vpaddd(dst, dst, vmmAxDimSum);
        add(regIdxIter, vlenYmm);
    cmp(regIdxIter, jcp_.indicesSize);
    jl(lExit, T_NEAR);
        uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
        mov(regIdxIter, 0);
    jmp(lExit, T_NEAR);
    L(lPerXmm);
        for (int i = 0; i < 2; i++) {
            fillIndicies(xmmAux0, xmmAux2);
            vinsertf128(dst, dst, xmmAux0, i);
            vinsertf128(mask, mask, xmmAux2, i);
        }
    L(lExit);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::vpGatherDD(const Xbyak::Ymm& dst) {
    fillIndicies(vmmSrcShifts, vmmGatherMask);
//mov(regAux1, ptr[regParams + GET_OFF(tmp)]);
//uni_vmovups(ptr[regAux1], vmmSrcShifts);
    vpgatherdd(dst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::vpGatherDD(const Xbyak::Zmm& dst) {
    fillIndicies(vmmSrcShifts, kMaskAux1);
    vpgatherdd(dst | kMaskAux1, ptr[regSrc + vmmSrcShifts]);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::vpGatherDDBlk(const Xbyak::Ymm& dst) {
    fillIndiciesBlk(vmmSrcShifts, vmmGatherMask);
    vpgatherdd(dst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::gatherAndGroup(const Xbyak::Ymm& dst, const Xbyak::Ymm& shufMask) {
    vpGatherDD(dst);
    vpshufb(dst, dst, shufMask);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::gatherAndGroup(const Xbyak::Zmm& dst, const Xbyak::Zmm& shufMask) {
    vpGatherDD(dst);
    vpshufb(dst | kMaskOnes, dst, shufMask);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::vpGatherDDlong(Xbyak::Ymm& dst) {
//fillIndiciesLong(dst, vmmGatherMask);
    fillIndiciesLong(vmmSrcShifts, vmmGatherMask);
//mov(regAux1, ptr[regParams + GET_OFF(tmp)]);
//uni_vmovups(ptr[regAux1], vmmSrcShifts);
    vpgatherdd(dst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
}