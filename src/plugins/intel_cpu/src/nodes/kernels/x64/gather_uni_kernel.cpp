// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_uni_kernel.hpp"

#include "openvino/core/except.hpp"

using namespace dnnl::impl::cpu;

namespace ov::intel_cpu {

const unsigned jitGatherKernelBase::shufMask8bitUni[16] = {0x0C080400,
                                                           0x80808080,
                                                           0x80808080,
                                                           0x80808080,
                                                           0x0C080400,
                                                           0x80808080,
                                                           0x80808080,
                                                           0x80808080,
                                                           0x0C080400,
                                                           0x80808080,
                                                           0x80808080,
                                                           0x80808080,
                                                           0x0C080400,
                                                           0x80808080,
                                                           0x80808080,
                                                           0x80808080};
const unsigned jitGatherKernelBase::permMask8bitA2[8] = {0, 4, 1, 5, 2, 6, 3, 7};
const unsigned jitGatherKernelBase::permMask8bitA5[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};

const unsigned jitGatherKernelBase::shufMask16bitUni[16] = {0x05040100,
                                                            0x0D0C0908,
                                                            0x80808080,
                                                            0x80808080,
                                                            0x05040100,
                                                            0x0D0C0908,
                                                            0x80808080,
                                                            0x80808080,
                                                            0x05040100,
                                                            0x0D0C0908,
                                                            0x80808080,
                                                            0x80808080,
                                                            0x05040100,
                                                            0x0D0C0908,
                                                            0x80808080,
                                                            0x80808080};
const unsigned jitGatherKernelBase::permMask16bitA2[8] = {0, 1, 4, 5, 2, 3, 6, 7};
const unsigned jitGatherKernelBase::permMask16bitA5[16] = {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15};

const unsigned jitGatherKernelBase::incVec[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

#define GET_OFF(field) offsetof(gatherJitExecArgs, field)

template <x64::cpu_isa_t isa>
jitUniGatherKernel<isa>::jitUniGatherKernel(const jGatherConfParams& jcp)
    : jitGatherKernelBase(jcp, x64::cpu_isa_traits<isa>::vlen, indicesTypeSize),
      x64::jit_generator(jit_name()) {
    if (jcp.dataTypeSize == 2) {
        dataTypeShift = 1;
    } else if (jcp.dataTypeSize == 4) {
        dataTypeShift = 2;
    }

    if (isa == x64::avx2) {
        permMask8bitUni = permMask8bitA2;
        permMask16bitUni = permMask16bitA2;
    } else if (isa == x64::avx512_core) {
        permMask8bitUni = permMask8bitA5;
        permMask16bitUni = permMask16bitA5;
    }
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::create_ker() {
    auto code = x64::jit_generator::create_kernel();
    if (code != dnnl::impl::status::success) {
        OPENVINO_THROW("Could not create Gather kernel. Error code: ", std::to_string(code));
    }
    ker_ = (decltype(ker_))jit_ker();
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::generate() {
    this->preamble();

    mov(regSrc, ptr[regParams + GET_OFF(src)]);
    mov(regDst, ptr[regParams + GET_OFF(dst)]);
    mov(regIndices, ptr[regParams + GET_OFF(indices)]);

    mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);

    auto& vAux0 = vmmAuxContainer[0];
    auto& vAux1 = vmmAuxContainer[1];
    auto& xAux0 = xmmAuxContainer[0];
    auto& xAux1 = xmmAuxContainer[1];

    uni_vpxor(vmmZeros, vmmZeros, vmmZeros);
    mov(regAux1, ptr[regParams + GET_OFF(axisDim)]);
    uni_vpbroadcastd(vmmAxisDim, ptr[regAux1]);

    if (!jcp.dynamicShapes) {
        mov(regAux1, ptr[regParams + GET_OFF(specIndicesSize)]);
        uni_vpbroadcastd(vmmSpecIdxSizeB, ptr[regAux1]);
        uni_vpslld(vmmSpecIdxSizeB, vmmSpecIdxSizeB, idxTypeShift);  // multiply by indices type size.

        mov(regAux1, ptr[regParams + GET_OFF(specIdxB)]);
        uni_vmovups(vmmSpecIdxB, ptr[regAux1]);

        if (jcp.beforeAxisSize != 1lu) {
            mov(regAux1, ptr[regParams + GET_OFF(dataBeforeAxisSumB)]);
            uni_vmovups(vmmSrcBeforeAxisSumB, ptr[regAux1]);
        }

        if (jcp.afterAxisSize == 1lu) {  // Elementwise case.
            uni_vmovd(reg32SpecIdxSizeB, xmmSpecIdxSizeB);
            if (jcp.beforeAxisSize != 1lu) {
                mov(regAux1, ptr[regParams + GET_OFF(axisAndAfterAxisSizeB)]);
                uni_vpbroadcastd(vmmAxisAndAfterAxisSizeB, ptr[regAux1]);
            }

            mov(regAux1, ptr[regParams + GET_OFF(idxBatchSumB)]);
            uni_vmovups(vmmIdxBatchSumB, ptr[regAux1]);

            mov(regAux1, ptr[regParams + GET_OFF(betweenBatchAndAxisSize)]);
            mov(regBetweenBatchAndAxisSize, ptr[regAux1]);
            mov(regBetweenBatchAndAxisIter, ptr[regParams + GET_OFF(betweenBatchAndAxisIter)]);

            if (jcp.specIdxSize < idxElPerVec) {  // Short case.
                if (jcp.specIdxSize != 1 && jcp.specIdxSize != 2 && jcp.specIdxSize != 4 && jcp.specIdxSize != 8 &&
                    jcp.specIdxSize != 16) {
                    mov(regAux1, ptr[regParams + GET_OFF(permIdxMask)]);
                    uni_vmovups(vmmPermIdxMask, ptr[regAux1]);
                }
                if (jcp.beforeAxisSize != 1lu) {
                    mov(regAux1, ptr[regParams + GET_OFF(beforeAxisDiff)]);
                    uni_vmovups(vmmBeforeAxDiffB, ptr[regAux1]);
                    if (jcp.dataTypeSize != 1) {
                        uni_vpslld(vmmBeforeAxDiffB, vmmBeforeAxDiffB, dataTypeShift);  // multiply by data type size
                    }
                }
                if (jcp.batchDims > 0lu) {
                    mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeB)]);
                    uni_vpbroadcastd(vmmSrcAfterBatchSizeB, ptr[regAux1]);
                }

                process(true, false);
            } else {  // Long case.
                uni_vmovd(reg32IdxIter, xmmSpecIdxB);
                fillVlenVector();

                process(false, false);
            }
        } else {                                     // Blocked case.
            if (jcp.afterAxisSize <= idxElPerVec) {  // Short case.
                mov(regAux1, ptr[regParams + GET_OFF(afterAxIdxB)]);
                uni_vmovups(vmmAfterAxisIdxB, ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(afterAxisPermMask)]);
                uni_vmovups(vmmAfterAxisPermMask, ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(specIdxDiff)]);
                uni_vmovups(vmmSpecIdxDiff, ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeB)]);
                uni_vpbroadcastd(vmmSrcAfterBatchSizeB, ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(afterAxisSize)]);
                uni_vpbroadcastd(vmmAfterAxisSize, ptr[regAux1]);

                if (jcp.beforeAxisSize != 1lu) {
                    mov(rSpecIdxAndAfterAxIterB, ptr[regParams + GET_OFF(specIdxAndAfterAxIterB)]);
                    mov(rSpecIdxAndAfterAxSizeB, ptr[regParams + GET_OFF(specIdxAndAfterAxSizeB)]);
                    if (jcp.specIdxSize * jcp.afterAxisSize < idxElPerVec) {
                        mov(regAux1, ptr[regParams + GET_OFF(beforeAxisDiff)]);
                        uni_vmovups(vmmBeforeAxDiffB, ptr[regAux1]);
                    } else {
                        mov(regAux1, ptr[regParams + GET_OFF(axisAndAfterAxisSizeB)]);
                        uni_vpbroadcastd(vmmAxisAndAfterAxisSizeB, ptr[regAux1]);
                    }
                    const uint64_t specIdxAndAfterAxisSize = jcp.specIdxSize * jcp.afterAxisSize;
                    if (specIdxAndAfterAxisSize != 1 && specIdxAndAfterAxisSize != 2 && specIdxAndAfterAxisSize != 4 &&
                        specIdxAndAfterAxisSize != 8 && specIdxAndAfterAxisSize != 16) {
                        mov(regAux1, ptr[regParams + GET_OFF(beforeAxisPermMask)]);
                        uni_vmovups(vmmBeforeAxPermMask, ptr[regAux1]);
                    }
                }

                process(true, true);
            } else {  // Long case.
                OPENVINO_THROW("Gather kernel does not support static shape with after axis size greater than elements "
                               "in vector.");
            }
        }
    } else {  // Dynamic shapes.
        mov(regAux1, ptr[regParams + GET_OFF(start)]);
        uni_vpbroadcastd(vmmSpecIdxB, ptr[regAux1]);
        mov(regAux1, reinterpret_cast<uintptr_t>(incVec));
        uni_vpaddd(vmmSpecIdxB, vmmSpecIdxB, ptr[regAux1]);
        vcvtdq2ps(vmmSpecIdxB, vmmSpecIdxB);

        // Formula: specIndices = (start % specIndicesSize) * idxTypeSize
        mov(regAux1, ptr[regParams + GET_OFF(specIndicesSize)]);
        uni_vpbroadcastd(vmmSpecIdxSizeB, ptr[regAux1]);
        uni_vcvtdq2ps(vAux1, vmmSpecIdxSizeB);
        uni_vdivps(vmmSrcBeforeAxisSumB, vmmSpecIdxB, vAux1);
        uni_vroundps(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, 0x1);
        uni_vfnmadd231ps(vmmSpecIdxB, vmmSrcBeforeAxisSumB, vAux1);
        uni_vcvtps2dq(vmmSpecIdxB, vmmSpecIdxB);
        uni_vpslld(vmmSpecIdxB, vmmSpecIdxB, idxTypeShift);          // multiply by indices type size.
        uni_vpslld(vmmSpecIdxSizeB, vmmSpecIdxSizeB, idxTypeShift);  // multiply by indices type size.
        uni_vmovd(reg32SpecIdxSizeB, xmmSpecIdxSizeB);

        mov(regAux1, ptr[regParams + GET_OFF(betweenBatchAndAxisSize)]);
        uni_vpbroadcastd(vAux1, ptr[regAux1]);
        uni_vmovd(reg32BetweenBatchAndAxisSize, xAux1);
        uni_vcvtdq2ps(vAux1, vAux1);
        uni_vdivps(vmmIdxBatchSumB, vmmSrcBeforeAxisSumB, vAux1);
        uni_vroundps(vmmIdxBatchSumB, vmmIdxBatchSumB, 0x1);
        uni_vfnmadd231ps(vmmSrcBeforeAxisSumB, vmmIdxBatchSumB, vAux1);
        uni_vcvtps2dq(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB);
        uni_vmovd(reg32BetweenBatchAndAxisIter, xmmSrcBeforeAxisSum);
        uni_vcvtps2dq(vmmIdxBatchSumB, vmmIdxBatchSumB);

        mov(regAux1, ptr[regParams + GET_OFF(axisAndAfterAxisSizeB)]);
        uni_vpbroadcastd(vmmAxisAndAfterAxisSizeB, ptr[regAux1]);
        // Formula: srcBeforeAxisSum = ((start / specIndicesSize) % betweenBatchAndAxis) * axisAndAfterAxisSize +
        // srcAfterBatchSize * idxBatchSum
        if (jcp.beforeAxisSize != 1lu) {
            uni_vpmulld(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vmmAxisAndAfterAxisSizeB);
            mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeB)]);
            uni_vpbroadcastd(vAux0, ptr[regAux1]);
            uni_vpmulld(vAux0, vAux0, vmmIdxBatchSumB);
            uni_vpaddd(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vAux0);
        }

        // Formula: idxBatchSum = specIdxSize * (start / afterBatchSize)
        uni_vpmulld(vmmIdxBatchSumB, vmmIdxBatchSumB, vmmSpecIdxSizeB);

        Xbyak::Label lBlock, lEnd;
        mov(regAux2, ptr[regParams + GET_OFF(afterAxSize)]);
        cmp(regAux2, 1);
        jg(lBlock, T_NEAR);
        {
            Xbyak::Label lLessThanVector1, lTail1, lTail2, lE1;

            cmp(regSpecIdxSizeB, vlen);
            jl(lLessThanVector1, T_NEAR);
            uni_vmovd(reg32IdxIter, xmmSpecIdxB);
            fillVlenVector();

            process(false, false);
            jmp(lE1, T_NEAR);
            L(lLessThanVector1);
            mov(regAux1, ptr[regParams + GET_OFF(permIdxMask)]);
            uni_vmovups(vmmPermIdxMask, ptr[regAux1]);
            if (jcp.beforeAxisSize != 1lu) {
                mov(regAux1, ptr[regParams + GET_OFF(beforeAxisDiff)]);
                uni_vmovups(vmmBeforeAxDiffB, ptr[regAux1]);
                if (jcp.dataTypeSize != 1) {
                    uni_vpslld(vmmBeforeAxDiffB, vmmBeforeAxDiffB, dataTypeShift);  // multiply by data type size
                }
            }
            mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeB)]);
            uni_vpbroadcastd(vmmSrcAfterBatchSizeB, ptr[regAux1]);

            process(true, false);
            L(lE1);
            jmp(lEnd, T_NEAR);
        }
        L(lBlock);
        {
            mov(regAux1, ptr[regParams + GET_OFF(start)]);
            uni_vpbroadcastd(vmmAfterAxisIdxB, ptr[regAux1]);
            mov(regAux1, reinterpret_cast<uintptr_t>(incVec));
            uni_vpaddd(vmmAfterAxisIdxB, vmmAfterAxisIdxB, ptr[regAux1]);
            uni_vcvtdq2ps(vmmAfterAxisIdxB, vmmAfterAxisIdxB);

            // afterAxIdxB = (start % afterAxSize) * idxTypeSize
            movd(xAux0, reg32Aux1);
            uni_vpbroadcastd(vAux1, xAux0);
            uni_vcvtdq2ps(vAux1, vAux1);
            uni_vdivps(vmmSrcBeforeAxisSumB, vmmAfterAxisIdxB, vAux1);
            uni_vroundps(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, 0x1);
            uni_vfnmadd231ps(vmmAfterAxisIdxB, vmmSrcBeforeAxisSumB, vAux1);
            uni_vcvtps2dq(vmmAfterAxisIdxB, vmmAfterAxisIdxB);
            uni_vpslld(vmmAfterAxisIdxB, vmmAfterAxisIdxB, idxTypeShift);  // multiply by indices type size.

            Xbyak::Label lLessThanVector2, lTail3, lTail4, lE2;

            cmp(regAux2, dataElPerVec);
            jl(lLessThanVector2, T_NEAR);
            uni_vmovd(reg32IdxIter, xmmSpecIdxB);
            fillVlenVector();

            //                process(false, true);
            jmp(lE2, T_NEAR);
            L(lLessThanVector2);
            auto& vAux2 = vmmAuxContainer[2];
            // Calculate permute mask
            uni_vmovd(xAux0, reg32Aux2);
            uni_vpbroadcastd(vAux1, xAux0);
            mov(regAux1, reinterpret_cast<uintptr_t>(&idxElPerVec));
            uni_vpbroadcastd(vAux0, ptr[regAux1]);
            uni_vpsubd(vmmAfterAxisPermMask, vAux0, vAux1);
            mov(regAux1, reinterpret_cast<uintptr_t>(incVec));
            uni_vpaddd(vmmAfterAxisPermMask, vmmAfterAxisPermMask, ptr[regAux1]);
            for (int i = 0; i < 6; i++) {
                if (isa == x64::avx512_core) {
                    auto kMask2 = Xbyak::Opmask(vAux2.getIdx());
                    vpcmpgtd(kMask2, vAux0, vmmAfterAxisPermMask);
                    uni_vpsubd(vmmAfterAxisPermMask | kMask2, vmmAfterAxisPermMask, vAux1);
                } else {
                    vpcmpgtd(vAux2, vAux0, vmmAfterAxisPermMask);
                    vpandn(vAux2, vAux2, vAux1);
                    uni_vpsubd(vmmAfterAxisPermMask, vmmAfterAxisPermMask, vAux2);
                }
            }

            process(true, true);
            L(lE2);
        }
        L(lEnd);
    }

    this->postamble();
}

template <>
void jitUniGatherKernel<x64::avx2>::uniVpGatherDd(Vmm& vDst, const Xbyak::Address& srcAddr, Vmask& kMask) {
    vpgatherdd(vDst, srcAddr, kMask);
}
template <>
void jitUniGatherKernel<x64::avx512_core>::uniVpGatherDd(Vmm& vDst, const Xbyak::Address& srcAddr, Vmask& kMask) {
    vpgatherdd(vDst | kMask, srcAddr);
}

template <>
void jitUniGatherKernel<x64::avx2>::normalizeRawIndices(Vmm& vRawIndices, Vmask& kDstMask, Vmask& kAuxMask) {
    // Compensate negative indices.
    if (jcp.reverseIndexing) {
        vpcmpgtd(kAuxMask, vmmZeros, vRawIndices);
        vpand(kAuxMask, kAuxMask, vmmAxisDim);
        uni_vpaddd(vRawIndices, vRawIndices, kAuxMask);
    }
    // Check boundaries.
    vpcmpgtd(kDstMask, vmmAxisDim, vRawIndices);
    vpcmpgtd(kAuxMask, vmmZeros, vRawIndices);
    vpandn(kDstMask, kAuxMask, kDstMask);
    // Multiply by type size.
    if (jcp.dataTypeSize > 1) {
        uni_vpslld(vRawIndices, vRawIndices, dataTypeShift);
    }
}

template <>
void jitUniGatherKernel<x64::avx512_core>::normalizeRawIndices(Vmm& vRawIndices, Vmask& kDstMask, Vmask& kAuxMask) {
    // Compensate negative indices.
    if (jcp.reverseIndexing) {
        vpcmpgtd(kAuxMask, vmmZeros, vRawIndices);
        uni_vpaddd(vRawIndices | kAuxMask, vRawIndices, vmmAxisDim);
    }
    // Check boundaries.
    vpcmpgtd(kAuxMask, vmmAxisDim, vRawIndices);
    vpcmpd(kDstMask | kAuxMask, vmmZeros, vRawIndices, 2);  // 2 - LE
    // Multiply by type size.
    if (jcp.dataTypeSize > 1) {
        uni_vpslld(vRawIndices, vRawIndices, dataTypeShift);
    }
}

template <>
void jitUniGatherKernel<x64::avx2>::normWithUpperBound(Vmm& vTarget, Vmm& vMax, Vmask& kAuxMask) {
    vpcmpgtd(kAuxMask, vMax, vTarget);
    vpandn(kAuxMask, kAuxMask, vMax);
    uni_vpsubd(vTarget, vTarget, kAuxMask);
}

template <>
void jitUniGatherKernel<x64::avx512_core>::normWithUpperBound(Vmm& vTarget, Vmm& vMax, Vmask& kAuxMask) {
    vpcmpd(kAuxMask, vMax, vTarget, 2);  // 2 -> LE
    uni_vpsubd(vTarget | kAuxMask, vTarget, vMax);
}

// Requires vAuxPool length 4.
// Returns calculated shifts in vAuxPool[0] and mask in vAuxPool[1].
template <>
void jitUniGatherKernel<x64::avx2>::calcSrcShiftLong(Vmm* vAuxPool, bool shiftFirst) {
    auto& vDstShifts = vAuxPool[0];
    auto& kDstMask = masksContainer[vAuxPool[1].getIdx()];
    auto& vAux0 = vAuxPool[2];
    auto& vAux1 = vAuxPool[3];
    auto& kAuxMask0 = masksContainer[vAux1.getIdx()];

    Xbyak::Label lIdxStride, lExit;
    if (shiftFirst) {
        uni_vpaddd(vmmSpecIdxB, vmmSpecIdxB, vmmVecLenB);
    }

    add(regIdxIter, vlen);
    cmp(regIdxIter, regSpecIdxSizeB);
    jge(lIdxStride, T_NEAR);
    if (jcp.batchDims > 0lu) {
        uni_vpaddd(vDstShifts, vmmIdxBatchSumB, vmmSpecIdxB);
        uni_vmovd(reg32Aux1, xmmAuxContainer[vDstShifts.getIdx()]);
    } else {
        uni_vmovd(reg32Aux1, xmmSpecIdxB);
    }
    vmovdqu(vDstShifts, ptr[regIndices + regAux1]);
    normalizeRawIndices(vDstShifts, kDstMask, kAuxMask0);
    if (jcp.beforeAxisSize != 1lu) {
        uni_vpaddd(vDstShifts, vDstShifts, vmmSrcBeforeAxisSumB);
    }
    jmp(lExit, T_NEAR);
    L(lIdxStride);
    sub(regIdxIter, regSpecIdxSizeB);
    vpcmpeqd(kDstMask, vAux0, vAux0);
    if (shiftFirst) {
        vpcmpgtd(vAux0, vmmSpecIdxSizeB, vmmSpecIdxB);
        vpandn(vAux1, vAux0, vmmSpecIdxSizeB);
        uni_vpsubd(vAux1, vmmSpecIdxB, vAux1);
        if (jcp.batchDims > 0lu) {
            uni_vpaddd(vAux1, vmmIdxBatchSumB, vAux1);
        }
        uni_vpsubd(vmmSpecIdxB, vmmSpecIdxB, vmmSpecIdxSizeB);
    } else {
        if (jcp.batchDims > 0lu) {
            uni_vpaddd(vAux0, vmmIdxBatchSumB, vmmSpecIdxB);
            uniVpGatherDd(vDstShifts, ptr[regIndices + vAux0], kDstMask);
        } else {
            uniVpGatherDd(vDstShifts, ptr[regIndices + vmmSpecIdxB], kDstMask);
        }
        normalizeRawIndices(vDstShifts, kDstMask, kAuxMask0);

        uni_vpbroadcastd(vAux0, xmmSpecIdxB);
        vpcmpgtd(vAux1, vAux0, vmmSpecIdxB);
        vpandn(vAux0, vAux1, vmmSpecIdxSizeB);
        uni_vpsubd(vmmSpecIdxB, vmmSpecIdxB, vAux0);

        if (jcp.beforeAxisSize != 1lu) {
            uni_vpaddd(vDstShifts, vDstShifts, vmmSrcBeforeAxisSumB);
            vpandn(vAux0, vAux1, vmmAxisAndAfterAxisSizeB);
            uni_vpaddd(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vAux0);
        }
    }

    if (jcp.batchDims > 0lu) {
        Xbyak::Label l1;
        inc(regBetweenBatchAndAxisIter);
        cmp(regBetweenBatchAndAxisIter, regBetweenBatchAndAxisSize);
        jl(l1, T_NEAR);
        mov(regBetweenBatchAndAxisIter, 0);
        if (shiftFirst) {
            uni_vpaddd(vmmIdxBatchSumB, vmmIdxBatchSumB, vmmSpecIdxSizeB);
            vpandn(vDstShifts, vAux0, vmmSpecIdxSizeB);
            uni_vpaddd(vAux1, vAux1, vDstShifts);
        } else {
            vpandn(vAux0, vAux1, vmmSpecIdxSizeB);
            uni_vpaddd(vmmIdxBatchSumB, vmmIdxBatchSumB, vAux0);
        }
        L(l1);
    }

    if (shiftFirst) {
        uniVpGatherDd(vDstShifts, ptr[regIndices + vAux1], kDstMask);
        normalizeRawIndices(vDstShifts, kDstMask, kAuxMask0);

        if (jcp.beforeAxisSize != 1lu) {
            vpandn(vAux0, vAux0, vmmAxisAndAfterAxisSizeB);
            uni_vpaddd(vAux0, vAux0, vmmSrcBeforeAxisSumB);
            uni_vpaddd(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vmmAxisAndAfterAxisSizeB);

            uni_vpaddd(vDstShifts, vDstShifts, vAux0);
        }
    }
    L(lExit);
}

// Requires vAuxPool length 4.
// Returns calculated shifts in vAuxPool[0] and mask in vAuxPool[1].
template <>
void jitUniGatherKernel<x64::avx512_core>::calcSrcShiftLong(Vmm* vAuxPool, bool shiftFirst) {
    auto& vDstShifts = vAuxPool[0];
    auto& kDstMask = masksContainer[vAuxPool[1].getIdx()];
    auto& vAux0 = vAuxPool[2];
    auto& vAux1 = vAuxPool[3];
    auto& kAuxMask0 = masksContainer[vAux1.getIdx()];
    auto& kAuxMask1 = masksContainer[vAux1.getIdx() + 1];

    Xbyak::Label lIdxStride, lExit;
    if (shiftFirst) {
        uni_vpaddd(vmmSpecIdxB, vmmSpecIdxB, vmmVecLenB);
    }

    add(regIdxIter, vlen);
    cmp(regIdxIter, regSpecIdxSizeB);
    jge(lIdxStride, T_NEAR);
    if (jcp.batchDims > 0lu) {
        uni_vpaddd(vDstShifts, vmmIdxBatchSumB, vmmSpecIdxB);
        uni_vmovd(reg32Aux1, xmmAuxContainer[vDstShifts.getIdx()]);
    } else {
        uni_vmovd(reg32Aux1, xmmSpecIdxB);
    }
    vmovdqu64(vDstShifts, ptr[regIndices + regAux1]);
    normalizeRawIndices(vDstShifts, kDstMask, kAuxMask0);
    if (jcp.beforeAxisSize != 1lu) {
        uni_vpaddd(vDstShifts, vDstShifts, vmmSrcBeforeAxisSumB);
    }
    jmp(lExit, T_NEAR);
    L(lIdxStride);
    sub(regIdxIter, regSpecIdxSizeB);
    vpcmpeqd(kDstMask, vDstShifts, vDstShifts);
    if (shiftFirst) {
        vpcmpd(kAuxMask1, vmmSpecIdxSizeB, vmmSpecIdxB, 2);  // 2 -> LE
        if (jcp.batchDims > 0lu) {
            uni_vpaddd(vAux1, vmmIdxBatchSumB, vmmSpecIdxB);
            uni_vpsubd(vAux1 | kAuxMask1, vAux1, vmmSpecIdxSizeB);
        } else {
            uni_vmovups(vAux1, vmmSpecIdxB);
            uni_vpsubd(vAux1 | kAuxMask1, vmmSpecIdxB, vmmSpecIdxSizeB);
        }
        uni_vpsubd(vmmSpecIdxB, vmmSpecIdxB, vmmSpecIdxSizeB);
    } else {
        if (jcp.batchDims > 0lu) {
            uni_vpaddd(vAux0, vmmIdxBatchSumB, vmmSpecIdxB);
            uniVpGatherDd(vDstShifts, ptr[regIndices + vAux0], kDstMask);
        } else {
            uniVpGatherDd(vDstShifts, ptr[regIndices + vmmSpecIdxB], kDstMask);
        }
        normalizeRawIndices(vDstShifts, kDstMask, kAuxMask0);

        uni_vpbroadcastd(vAux0, xmmSpecIdxB);
        vpcmpd(kAuxMask1, vAux0, vmmSpecIdxB, 2);  // 2 -> LE
        uni_vpsubd(vmmSpecIdxB | kAuxMask1, vmmSpecIdxB, vmmSpecIdxSizeB);

        if (jcp.beforeAxisSize != 1lu) {
            uni_vpaddd(vDstShifts, vDstShifts, vmmSrcBeforeAxisSumB);
            uni_vpaddd(vmmSrcBeforeAxisSumB | kAuxMask1, vmmSrcBeforeAxisSumB, vmmAxisAndAfterAxisSizeB);
        }
    }

    if (jcp.batchDims > 0lu) {
        Xbyak::Label l1;
        inc(regBetweenBatchAndAxisIter);
        cmp(regBetweenBatchAndAxisIter, regBetweenBatchAndAxisSize);
        jl(l1, T_NEAR);
        mov(regBetweenBatchAndAxisIter, 0);
        if (shiftFirst) {
            uni_vpaddd(vmmIdxBatchSumB, vmmIdxBatchSumB, vmmSpecIdxSizeB);
            uni_vpaddd(vAux1 | kAuxMask1, vAux1, vmmSpecIdxSizeB);
        } else {
            uni_vpaddd(vmmIdxBatchSumB | kAuxMask1, vmmIdxBatchSumB, vmmSpecIdxSizeB);
        }
        L(l1);
    }

    if (shiftFirst) {
        uniVpGatherDd(vDstShifts, ptr[regIndices + vAux1], kDstMask);
        normalizeRawIndices(vDstShifts, kDstMask, kAuxMask0);

        if (jcp.beforeAxisSize != 1lu) {
            uni_vpaddd(vDstShifts, vDstShifts, vmmSrcBeforeAxisSumB);
            uni_vpaddd(vDstShifts | kAuxMask1, vDstShifts, vmmAxisAndAfterAxisSizeB);
            uni_vpaddd(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vmmAxisAndAfterAxisSizeB);
        }
    }
    L(lExit);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::calcSrcShiftLongBlock(Vmm* vAuxPool, bool shiftFirst) {
    // Most likely there will no significant performance gain vs memcpy in reference implementation on big blocks after
    // axis, therefore no time was invested to this case yet.
    OPENVINO_THROW("Unsupported case.");
}

// Requires vAuxPool length 3.
// Returns calculated shifts in vAuxPool[0] and mask in vAuxPool[1].
template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::calcSrcShiftShort(Vmm* vAuxPool, bool shiftFirst) {
    auto& vDstShifts = vAuxPool[0];
    auto& kDstMask = masksContainer[vAuxPool[1].getIdx()];
    auto& vAux0 = vAuxPool[2];

    if (shiftFirst) {
        if (jcp.beforeAxisSize != 1lu) {
            uni_vpaddd(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vmmBeforeAxDiffB);
        }
        // No sense to permute if specIdxSize is one of {1, 2, 4, 8, 16}. 0 is reserved for dynamic case.
        if (jcp.specIdxSize != 1 && jcp.specIdxSize != 2 && jcp.specIdxSize != 4 && jcp.specIdxSize != 8 &&
            jcp.specIdxSize != 16) {
            vpermd(vmmSpecIdxB, vmmPermIdxMask, vmmSpecIdxB);
            if (jcp.beforeAxisSize != 1lu) {
                vpermd(vmmBeforeAxDiffB, vmmPermIdxMask, vmmBeforeAxDiffB);
            }
        }
    }

    vpcmpeqd(kDstMask, vAux0, vAux0);
    if (jcp.batchDims > 0lu) {
        // Calculate indices batch sum.
        uni_vcvtdq2ps(vAux0, vmmSrcBeforeAxisSumB);
        uni_vcvtdq2ps(vDstShifts, vmmSrcAfterBatchSizeB);
        uni_vdivps(vAux0, vAux0, vDstShifts);
        uni_vroundps(vAux0, vAux0, 0x1);
        uni_vcvtps2dq(vAux0, vAux0);

        uni_vpmulld(vAux0, vAux0, vmmSpecIdxSizeB);
        uni_vpaddd(vAux0, vAux0, vmmSpecIdxB);

        uniVpGatherDd(vDstShifts, ptr[regIndices + vAux0], kDstMask);
    } else {
        uniVpGatherDd(vDstShifts, ptr[regIndices + vmmSpecIdxB], kDstMask);
    }

    auto& kAuxMask0 = masksContainer[vAux0.getIdx()];
    normalizeRawIndices(vDstShifts, kDstMask, kAuxMask0);
    if (jcp.beforeAxisSize != 1lu) {
        uni_vpaddd(vDstShifts, vDstShifts, vmmSrcBeforeAxisSumB);
    }
}

// Requires vAuxPool length 4.
// Returns calculated shifts in vAuxPool[0] and mask in vAuxPool[1].
template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::calcSrcShiftShortBlock(Vmm* vAuxPool, bool shiftFirst) {
    auto& vDstShifts = vAuxPool[0];
    auto& kDstMask = masksContainer[vAuxPool[1].getIdx()];
    auto& vAux0 = vAuxPool[2];
    auto& vAux1 = vAuxPool[3];
    auto& kAuxMask0 = masksContainer[vAux0.getIdx()];
    const uint64_t specIdxAndAfterAxisSize = jcp.specIdxSize * jcp.afterAxisSize;

    if (shiftFirst) {
        if (jcp.specIdxSize != 1) {
            uni_vpaddd(vmmSpecIdxB, vmmSpecIdxB, vmmSpecIdxDiff);
            normWithUpperBound(vmmSpecIdxB, vmmSpecIdxSizeB, kAuxMask0);
        }
        // No sense to permute if afterAxisSize is one of {1, 2, 4, 8, 16}. 0 is reserved for dynamic case.
        if (jcp.afterAxisSize != 1 && jcp.afterAxisSize != 2 && jcp.afterAxisSize != 4 && jcp.afterAxisSize != 8 &&
            jcp.afterAxisSize != 16) {
            vpermd(vmmAfterAxisIdxB, vmmAfterAxisPermMask, vmmAfterAxisIdxB);
            if (jcp.specIdxSize != 1) {
                vpermd(vmmSpecIdxDiff, vmmAfterAxisPermMask, vmmSpecIdxDiff);
            }
        }

        if (jcp.beforeAxisSize != 1lu) {
            if (!jcp.dynamicShapes) {
                if (specIdxAndAfterAxisSize > 0lu && specIdxAndAfterAxisSize <= idxElPerVec) {
                    uni_vpaddd(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vmmBeforeAxDiffB);
                    uni_vmovups(vAux1, vmmSrcBeforeAxisSumB);
                    if (specIdxAndAfterAxisSize != 1 && specIdxAndAfterAxisSize != 2 && specIdxAndAfterAxisSize != 4 &&
                        specIdxAndAfterAxisSize != 8 && specIdxAndAfterAxisSize != 16) {
                        vpermd(vmmBeforeAxDiffB, vmmBeforeAxPermMask, vmmBeforeAxDiffB);
                    }
                } else {
                    Xbyak::Label lBeforeAxStep, lBeforeAxStepEnd;
                    add(rSpecIdxAndAfterAxIterB, idxElPerVec * jcp.dataTypeSize);
                    cmp(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);
                    jl(lBeforeAxStep, T_NEAR);
                    sub(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);

                    vpmulld(vAux0, vmmSpecIdxB, vmmAfterAxisSize);
                    uni_vpaddd(vAux0, vAux0, vmmAfterAxisIdxB);
                    Xbyak::Xmm& xAux0 = xmmAuxContainer[vAux0.getIdx()];
                    uni_vpbroadcastd(vAux1, xAux0);
                    if (isa == x64::avx512_core) {
                        auto kMask0 = Xbyak::Opmask(kAuxMask0.getIdx());
                        vpcmpgtd(kMask0, vAux1, vAux0);
                        uni_vmovups(vAux1, vmmSrcBeforeAxisSumB);
                        uni_vpaddd(vAux1 | kMask0, vmmSrcBeforeAxisSumB, vmmAxisAndAfterAxisSizeB);
                    } else {
                        vpcmpgtd(vAux1, vAux1, vAux0);
                        vpand(vAux1, vAux1, vmmAxisAndAfterAxisSizeB);
                        uni_vpaddd(vAux1, vmmSrcBeforeAxisSumB, vAux1);
                    }
                    uni_vpaddd(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vmmAxisAndAfterAxisSizeB);
                    jmp(lBeforeAxStepEnd);
                    L(lBeforeAxStep);
                    uni_vmovups(vAux1, vmmSrcBeforeAxisSumB);
                    L(lBeforeAxStepEnd);
                }
            } else {
            }
        }
    } else {
        if (jcp.beforeAxisSize != 1lu) {
            uni_vmovups(vAux1, vmmSrcBeforeAxisSumB);
            if (specIdxAndAfterAxisSize > idxElPerVec) {
                // Broadcast the last element.
                if (isa == x64::avx512_core) {
                    vshuff64x2(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, 0xFF);
                } else {
                    vpermq(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, 0xFF);
                }
                vpshufd(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, 0xFF);

                Xbyak::Label lBeforeAxStepEnd1;
                add(rSpecIdxAndAfterAxIterB, idxElPerVec * jcp.dataTypeSize);
                cmp(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);
                jl(lBeforeAxStepEnd1, T_NEAR);
                sub(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);
                cmp(rSpecIdxAndAfterAxIterB, 0);
                jne(lBeforeAxStepEnd1, T_NEAR);
                uni_vpaddd(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vmmAxisAndAfterAxisSizeB);
                L(lBeforeAxStepEnd1);
            }
        }
    }

    vpcmpeqd(kDstMask, vAux0, vAux0);
    if (jcp.batchDims > 0lu) {
        // Calculate indices batch sum.
        uni_vcvtdq2ps(vAux0, vAux1);
        uni_vcvtdq2ps(vDstShifts, vmmSrcAfterBatchSizeB);
        uni_vdivps(vAux0, vAux0, vDstShifts);
        uni_vroundps(vAux0, vAux0, 0x1);
        uni_vcvtps2dq(vAux0, vAux0);

        uni_vpmulld(vAux0, vAux0, vmmSpecIdxSizeB);
        uni_vpaddd(vAux0, vAux0, vmmSpecIdxB);

        uniVpGatherDd(vDstShifts, ptr[regIndices + vAux0], kDstMask);
    } else {
        uniVpGatherDd(vDstShifts, ptr[regIndices + vmmSpecIdxB], kDstMask);
    }

    normalizeRawIndices(vDstShifts, kDstMask, kAuxMask0);

    if (jcp.afterAxisSize != 1lu) {
        vpmulld(vDstShifts, vDstShifts, vmmAfterAxisSize);
        uni_vpaddd(vDstShifts, vDstShifts, vmmAfterAxisIdxB);
    }
    if (jcp.beforeAxisSize != 1lu) {
        uni_vpaddd(vDstShifts, vDstShifts, vAux1);
    }
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::process(bool isShortIdx, bool blocked) {
    Xbyak::Label lTailProc, lEndProc;
    cmp(regWorkAmount, dataElPerVec);
    jl(lTailProc, T_NEAR);
    if (jcp.dataTypeSize == 4) {
        process32b(isShortIdx, blocked);
    } else if (jcp.dataTypeSize == 2) {
        process16b(isShortIdx, blocked);
    } else if (jcp.dataTypeSize == 1) {
        process8b(isShortIdx, blocked);
    }
    jmp(lEndProc, T_NEAR);
    L(lTailProc);
    tail(isShortIdx, false, blocked);
    L(lEndProc);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::process32b(bool isShortIdx, bool blocked) {
    Xbyak::Label lDstIdxLoop, lTail;

    // First iteration
    shiftIdxAndGather(vmmAuxContainer, isShortIdx, false, blocked);
    uni_vmovups(ptr[regDst], vmmAuxContainer[2]);

    // Main loop
    L(lDstIdxLoop);
    {
        add(regDst, vlen);
        sub(regWorkAmount, dataElPerVec);
        cmp(regWorkAmount, dataElPerVec);
        jl(lTail, T_NEAR);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        uni_vmovups(ptr[regDst], vmmAuxContainer[2]);

        jmp(lDstIdxLoop, T_NEAR);
    }

    L(lTail);
    tail(isShortIdx, true, blocked);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::process16b(bool isShortIdx, bool blocked) {
    Xbyak::Label lDstIdxLoop1, lTail;

    Vmm vShufMask, vPermMask, vBuff0;
    if (isa == x64::avx512_core) {
        vPermMask = vmmAuxContainer[7];
        vShufMask = vmmAuxContainer[8];
        vBuff0 = vmmAuxContainer[9];
    } else {
        vPermMask = vmmAuxContainer[1];
        vShufMask = vmmAuxContainer[4];
        vBuff0 = vmmAuxContainer[5];
    }

    mov(regAux1, reinterpret_cast<uintptr_t>(shufMask16bitUni));
    uni_vmovups(vShufMask, ptr[regAux1]);

    // First iteration
    shiftIdxAndGather(vmmAuxContainer, isShortIdx, false, blocked);
    vpshufb(vBuff0, vmmAuxContainer[2], vShufMask);

    shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
    vpshufb(vmmAuxContainer[0], vmmAuxContainer[2], vShufMask);

    vshufps(vmmAuxContainer[0], vBuff0, vmmAuxContainer[0], 0x44);
    // vPermMask(vmm1) is override in shiftIdxAndGather, load the mask here for correctness
    mov(regAux1, reinterpret_cast<uintptr_t>(permMask16bitUni));
    uni_vmovups(vPermMask, ptr[regAux1]);
    vpermd(vmmAuxContainer[0], vPermMask, vmmAuxContainer[0]);

    uni_vmovups(ptr[regDst], vmmAuxContainer[0]);

    // Main loop.
    L(lDstIdxLoop1);
    {
        add(regDst, vlen);
        sub(regWorkAmount, dataElPerVec);
        cmp(regWorkAmount, dataElPerVec);
        jl(lTail, T_NEAR);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        vpshufb(vBuff0, vmmAuxContainer[2], vShufMask);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        vpshufb(vmmAuxContainer[0], vmmAuxContainer[2], vShufMask);

        vshufps(vmmAuxContainer[0], vBuff0, vmmAuxContainer[0], 0x44);
        if (isa == x64::avx2) {
            // Register vPermMask is invalidated by shiftIdxAndGather and must be initialized again.
            mov(regAux1, reinterpret_cast<uintptr_t>(permMask16bitUni));
            uni_vmovups(vPermMask, ptr[regAux1]);
        }
        vpermd(vmmAuxContainer[0], vPermMask, vmmAuxContainer[0]);

        uni_vmovups(ptr[regDst], vmmAuxContainer[0]);

        jmp(lDstIdxLoop1, T_NEAR);
    }

    L(lTail);
    tail(isShortIdx, true, blocked);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::process8b(bool isShortIdx, bool blocked) {
    Xbyak::Label lDstIdxLoop1, lTail;

    Vmm vShufMask, vPermMask, vBuff0, vBuff1;
    if (isa == x64::avx512_core) {
        vPermMask = vmmAuxContainer[7];
        vShufMask = vmmAuxContainer[8];
        vBuff0 = vmmAuxContainer[9];
        vBuff1 = vmmAuxContainer[10];
    } else {
        vPermMask = vmmAuxContainer[1];
        vShufMask = vmmAuxContainer[4];
        vBuff0 = vmmAuxContainer[5];
        vBuff1 = vmmAuxContainer[6];
    }
    mov(regAux1, reinterpret_cast<uintptr_t>(shufMask8bitUni));
    uni_vmovups(vShufMask, ptr[regAux1]);

    // First iteration
    shiftIdxAndGather(vmmAuxContainer, isShortIdx, false, blocked);
    vpshufb(vBuff0, vmmAuxContainer[2], vShufMask);

    shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
    vpshufb(vmmAuxContainer[0], vmmAuxContainer[2], vShufMask);

    vshufps(vBuff0, vBuff0, vmmAuxContainer[0], 0x0);

    shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
    vpshufb(vBuff1, vmmAuxContainer[2], vShufMask);

    shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
    vpshufb(vmmAuxContainer[0], vmmAuxContainer[2], vShufMask);

    vshufps(vBuff1, vBuff1, vmmAuxContainer[0], 0x0);
    vshufps(vmmAuxContainer[0], vBuff0, vBuff1, 0x88);

    mov(regAux1, reinterpret_cast<uintptr_t>(permMask8bitUni));
    uni_vmovups(vPermMask, ptr[regAux1]);

    vpermd(vmmAuxContainer[0], vPermMask, vmmAuxContainer[0]);

    uni_vmovups(ptr[regDst], vmmAuxContainer[0]);

    // Main loop.
    L(lDstIdxLoop1);
    {
        add(regDst, vlen);
        sub(regWorkAmount, dataElPerVec);
        cmp(regWorkAmount, dataElPerVec);
        jl(lTail, T_NEAR);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        vpshufb(vBuff0, vmmAuxContainer[2], vShufMask);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        vpshufb(vmmAuxContainer[0], vmmAuxContainer[2], vShufMask);

        vshufps(vBuff0, vBuff0, vmmAuxContainer[0], 0x0);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        vpshufb(vBuff1, vmmAuxContainer[2], vShufMask);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        vpshufb(vmmAuxContainer[0], vmmAuxContainer[2], vShufMask);

        vshufps(vmmAuxContainer[0], vBuff1, vmmAuxContainer[0], 0x0);
        vshufps(vmmAuxContainer[0], vBuff0, vmmAuxContainer[0], 0x88);

        if (isa == x64::avx2) {
            // Register vPermMask is invalidated by shiftIdxAndGather and must be initialized again.
            mov(regAux1, reinterpret_cast<uintptr_t>(permMask8bitUni));
            uni_vmovups(vPermMask, ptr[regAux1]);
        }
        vpermd(vmmAuxContainer[0], vPermMask, vmmAuxContainer[0]);

        uni_vmovups(ptr[regDst], vmmAuxContainer[0]);

        jmp(lDstIdxLoop1, T_NEAR);
    }

    L(lTail);
    tail(isShortIdx, true, blocked);
}

// Requires vAuxPool length 4.
// Returns gathered data in vAuxPool[2].
template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::shiftIdxAndGather(Vmm* vAuxPool, bool isShortIdx, bool shiftFirst, bool blocked) {
    if (blocked) {
        if (isShortIdx) {
            calcSrcShiftShortBlock(vAuxPool, shiftFirst);
        } else {
            calcSrcShiftLongBlock(vAuxPool, shiftFirst);
        }
    } else {
        if (isShortIdx) {
            calcSrcShiftShort(vAuxPool, shiftFirst);
        } else {
            calcSrcShiftLong(vAuxPool, shiftFirst);
        }
    }
    auto& kGatherMask = masksContainer[vAuxPool[1].getIdx()];
    uni_vmovups(vAuxPool[2], vmmZeros);
    uniVpGatherDd(vAuxPool[2], ptr[regSrc + vAuxPool[0]], kGatherMask);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::tail(bool isShortIdx, bool shiftFirst, bool blocked) {
    auto& vSrcShift = vmmAuxContainer[0];
    auto& kGatherMask = masksContainer[vmmAuxContainer[1].getIdx()];
    auto& vAux0 = vmmAuxContainer[2];
    auto& vAux1 = vmmAuxContainer[3];
    auto& kAuxMask1 = masksContainer[vAux1.getIdx()];
    Xbyak::Label lEnd;

    const int secondStepCycles = 4 / jcp.dataTypeSize;
    for (int p = 0; p < secondStepCycles; p++) {
        cmp(regWorkAmount, 0);
        jle(lEnd, T_NEAR);

        if (isShortIdx) {
            if (blocked) {
                calcSrcShiftShortBlock(vmmAuxContainer, p > 0 || shiftFirst);
            } else {
                calcSrcShiftShort(vmmAuxContainer, p > 0 || shiftFirst);
            }
        } else {
            if (blocked) {
                calcSrcShiftLongBlock(vmmAuxContainer, p > 0 || shiftFirst);
            } else {
                calcSrcShiftLong(vmmAuxContainer, p > 0 || shiftFirst);
            }
        }

        fillRestWorkMask(kAuxMask1, vAux0, regWorkAmount, regAux1, rdx);

        // Combining masks.
        if (isa == x64::avx512_core) {
            auto kMask1 = Xbyak::Opmask(kAuxMask1.getIdx());
            auto kMaskG = Xbyak::Opmask(kGatherMask.getIdx());
            kandd(kMaskG, kMaskG, kMask1);
        } else if (isa == x64::avx2) {
            auto& vGatherMask = vmmAuxContainer[kGatherMask.getIdx()];
            vpand(vGatherMask, vGatherMask, vAux1);
        }

        uni_vmovups(vAux0, vmmZeros);
        uniVpGatherDd(vAux0, ptr[regSrc + vSrcShift], kGatherMask);
        if (jcp.dataTypeSize == 4) {
            uni_vmovups_tail(ptr[regDst], kAuxMask1, vAux0);
            sub(regWorkAmount, dataElPerVec);
        } else {
            storeVectorPart(regDst, regWorkAmount, vAux0, vAux1);
        }
    }
    L(lEnd);
}

template <>
void jitUniGatherKernel<x64::avx512_core>::fillRestWorkMask(Vmask& kDstMask,
                                                            Vmm& vmmAux,
                                                            const Xbyak::Reg64& rWorkRest,
                                                            const Xbyak::Reg64& rAux0,
                                                            const Xbyak::Reg64& rAux1) {
    Xbyak::Label lKmov;
    Xbyak::Reg32 rOnes(rAux1.getIdx());
    mov(rOnes, 0x0000FFFF);
    cmp(rWorkRest, idxElPerVec);
    jge(lKmov);
    Xbyak::Reg8 rShift(Xbyak::Operand::CL);
    mov(rShift, idxElPerVec);
    sub(rShift, rWorkRest);
    shr(rOnes, rShift);
    L(lKmov);
    kmovw(kDstMask, rOnes);
}

template <>
void jitUniGatherKernel<x64::avx2>::fillRestWorkMask(Vmask& kDstMask,
                                                     Vmm& vAux,
                                                     const Xbyak::Reg64& rWorkRest,
                                                     const Xbyak::Reg64& rAux0,
                                                     const Xbyak::Reg64& rAux1) {
    Xbyak::Label lEnd;
    mov(rAux0, rWorkRest);
    Xbyak::Reg32 rOnes(rAux1.getIdx());
    mov(rOnes, 0xFFFFFFFF);
    Xbyak::Xmm xmmAux(vAux.getIdx());
    uni_vmovups(kDstMask, vmmZeros);
    for (size_t i = 0; i < idxElPerVec; i++) {
        cmp(rAux0, 0);
        je(lEnd, T_NEAR);

        if (i % 4 == 0) {
            uni_vmovups(xmmAux, xmmZeros);
        }

        vpinsrd(xmmAux, xmmAux, rOnes, i % 4);
        vinserti128(kDstMask, kDstMask, xmmAux, i / 4);
        sub(rAux0, 1);
    }
    L(lEnd);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::storeVectorPart(const Xbyak::Reg64& rDst,
                                              const Xbyak::Reg64& rToStoreCounter,
                                              Vmm& vmmSrc,
                                              Vmm& vAux) {
    Xbyak::Label lEnd;
    Xbyak::Xmm xAux(vAux.getIdx());
    for (size_t j = 0; j < vlen / vlenXmm; j++) {
        if (isa == x64::avx2) {
            vextracti128(xAux, vmmSrc, j);
        } else if (isa == x64::avx512_core) {
            vextracti64x2(xAux, vmmSrc, j);
        }

        for (int k = 0; k < 4; k++) {
            cmp(rToStoreCounter, 0);
            jle(lEnd, T_NEAR);

            if (jcp.dataTypeSize == 4) {
                uni_vpextrd(ptr[rDst], xAux, k);
            } else if (jcp.dataTypeSize == 2) {
                uni_vpextrw(ptr[rDst], xAux, k * 2);
            } else if (jcp.dataTypeSize == 1) {
                uni_vpextrb(ptr[rDst], xAux, k * 4);
            }

            add(rDst, jcp.dataTypeSize);
            sub(rToStoreCounter, 1);
        }
    }
    L(lEnd);
}

template <>
void jitUniGatherKernel<x64::avx512_core>::fillVlenVector() {
    mov(reg32Aux1, vlen);
    vpbroadcastd(vmmVecLenB, reg32Aux1);
}
template <>
void jitUniGatherKernel<x64::avx2>::fillVlenVector() {
    vpcmpeqd(vmmVecLenB, vmmVecLenB, vmmVecLenB);
    vpsrld(vmmVecLenB, vmmVecLenB, 31);     // Right shift to 1.
    uni_vpslld(vmmVecLenB, vmmVecLenB, 5);  // Left shift to 32.
}

template <x64::cpu_isa_t isa>
bool jitUniGatherKernel<isa>::isSupportedConfiguration(uint64_t afterAxisSize) {
    if (!jcp.dynamicShapes && afterAxisSize <= idxElPerVec) {
        if (afterAxisSize > 1 && isa == x64::avx2 && (jcp.dataTypeSize == 1 || jcp.dataTypeSize == 2)) {
            // There are no enough registers for these cases.
            return false;
        }

        return true;
    }
    if (jcp.dynamicShapes && afterAxisSize == 1) {
        return true;
    }
    return false;
}

template struct jitUniGatherKernel<x64::avx2>;
template struct jitUniGatherKernel<x64::avx512_core>;

}  // namespace ov::intel_cpu
