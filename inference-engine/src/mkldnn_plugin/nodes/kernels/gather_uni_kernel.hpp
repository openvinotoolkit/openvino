// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/jit_generator.hpp"
#include <mkldnn_types.h>

namespace MKLDNNPlugin {

struct jGatherConfParams {
    uint64_t dataTypeSize = 1;
    bool reverseIndexing = true;
};

struct gatherJitExecArgs {
    const void* src;
    void* dst;
    const int* indices;
    const int* axisDim;
    const uint64_t* start;
    const uint64_t* specIndicesSize;
    const uint64_t* betweenBatchAndAxisSize;
    const uint64_t* axisAndAfterAxisSizeInBytes;
    const uint64_t* srcAfterBatchSizeInBytes;
    const int* permIdx;
    const int* beforeAxisDiff;
    uint64_t workAmount = 0;
    uint64_t afterAxisBlockSize = 0;
};

struct jitGatherKernelBase {
    void (*ker_)(const gatherJitExecArgs *);
    void operator()(const gatherJitExecArgs *args) {
        assert(ker_);
        ker_(args);
    }
    explicit jitGatherKernelBase(jGatherConfParams jcp) : ker_(nullptr), jcp(jcp) {}
    virtual ~jitGatherKernelBase() {}

    virtual void create_ker() = 0;
    inline uint64_t getVecLen() {
        return vlen;
    }
    inline uint64_t getDataElPerVec() {
        return dataElPerVec;
    }
    inline uint64_t getIdxElPerVec() {
        return idxElPerVec;
    }

protected:
    jGatherConfParams jcp;
    uint64_t vlen;
    uint64_t dataElPerVec;
    uint64_t idxElPerVec;
};

template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
struct jitUniGatherKernel : public jitGatherKernelBase, public mkldnn::impl::cpu::x64::jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jitUniGatherKernel)

    explicit jitUniGatherKernel(jGatherConfParams jcp);

    void create_ker() override;
    void generate() override;

protected:
    using Vmm = typename mkldnn::impl::utils::conditional<isa == mkldnn::impl::cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using Vmask = typename mkldnn::impl::utils::conditional<isa == mkldnn::impl::cpu::x64::avx2, Xbyak::Ymm, Xbyak::Opmask>::type;
    const uint32_t vlenXmm = mkldnn::impl::cpu::x64::cpu_isa_traits<mkldnn::impl::cpu::x64::sse41>::vlen;
    uint32_t dataTypeShift = 0;
    const uint32_t indicesTypeSize = sizeof(int);

    // 64b registers.
    Xbyak::Reg64 regSrc = r8;
    Xbyak::Reg64 regDst = r9;
    Xbyak::Reg64 regIndices = r10;
    Xbyak::Reg64 regIdxIter = r11;
    Xbyak::Reg64 regWorkAmount = r12;
    Xbyak::Reg64 regSpecIdxSizeInBytes = r13;
    Xbyak::Reg64 regAux1 = r14;
    Xbyak::Reg64 regAux2 = r15;
    Xbyak::Reg64 regBetweenBatchAndAxisIter = r15;
    Xbyak::Reg64 regBetweenBatchAndAxisSize = rbx;

    Xbyak::Reg64 regParams = mkldnn::impl::cpu::x64::abi_param1;

    // 32b registers.
    Xbyak::Reg32 reg32IdxIter = Xbyak::Reg32(regIdxIter.getIdx());
    Xbyak::Reg32 reg32SpecIdxSizeInBytes = Xbyak::Reg32(regSpecIdxSizeInBytes.getIdx());
    Xbyak::Reg32 reg32BetweenBatchAndAxisSize = Xbyak::Reg32(regBetweenBatchAndAxisSize.getIdx());
    Xbyak::Reg32 reg32BetweenBatchAndAxisIter = Xbyak::Reg32(regBetweenBatchAndAxisIter.getIdx());
    Xbyak::Reg32 reg32Aux1 = Xbyak::Reg32(regAux1.getIdx());

    // Opmasks.
    Xbyak::Opmask kMaskOnes = Xbyak::Opmask(1);
    Xbyak::Opmask kMaskAux0 = Xbyak::Opmask(2);
    Xbyak::Opmask kMaskAux1 = Xbyak::Opmask(3);
    Xbyak::Opmask kMaskAux2 = Xbyak::Opmask(4);
    Xbyak::Opmask kGatherMask = Xbyak::Opmask(5);

    Vmask vGatherMask;
    Vmask vAuxMask0;
    Vmask vAuxMask1;

    Vmm vmmAxisAndAfterAxisSize = Vmm(2);
    Vmm vmmSrcAfterBatchSize = Vmm(2);
    Vmm vmmBeforeAxisSum = Vmm(3);
    Vmm vmmSrcShifts = Vmm(5);
    Vmm vmmZeros = Vmm(6);
    Vmm vmmPermIdx = Vmm(8);
    // Common.
    Vmm vmmSpecIndicesInBytes = Vmm(9);
    Vmm vmmGatherMask = Vmm(11);
    Vmm vmmSpecIdxSizeInBytes = Vmm(12);
    Vmm vmmAxisDim = Vmm(14);
    Vmm vmmDst = Vmm(15);
    // Auxiliary.
    Vmm vmmAux0 = Vmm(0);
    Vmm vmmAux1 = Vmm(1);
    Vmm vmmAux3 = Vmm(8);
    Vmm vmmAux4 = Vmm(4);
    Vmm vmmAux6 = Vmm(13);
    // AVX512
    Vmm vmmAux5 = Vmm(16);

    // Only long.
    Vmm vmmVecLen = Vmm(7);
    Vmm vmmIdxBatchSum = Vmm(10);
    // Only short.
    Vmm vmmAux8 = Vmm(7);
    Vmm vmmBeforeAxisDiff = Vmm(10);

    // XMM
    Xbyak::Xmm xmmAux0 = Xbyak::Xmm(vmmAux0.getIdx());
    Xbyak::Xmm xmmAux1 = Xbyak::Xmm(vmmAux1.getIdx());
    Xbyak::Xmm xmmZeros = Xbyak::Xmm(vmmZeros.getIdx());
    Xbyak::Xmm xmmBeforeAxisSum = Xbyak::Xmm(vmmBeforeAxisSum.getIdx());
    Xbyak::Xmm xmmSpecIdxSizeInBytes = Xbyak::Xmm(vmmSpecIdxSizeInBytes.getIdx());
    Xbyak::Xmm xmmSpecIndicesInBytes = Xbyak::Xmm(vmmSpecIndicesInBytes.getIdx());

    void calcSrcShiftLong(Vmm& dstIndices, Vmask& dstMask, bool shiftFirst = true);
    void calcSrcShiftShort(Vmm& dstIndices, Vmask& dstMask, bool shiftFirst = true);
    void normalizeRawIndices(Vmm& rawIndices, Vmask& dstMask, Vmask& aux);
    void process32b(bool isShortIdx);
    void process16b(bool isShortIdx);
    void process8b(bool isShortIdx);
    void tail(bool isShortIdx, bool shiftFirst = true);
    void fillRestWorkMask(Vmm& vmmMask, Vmm& vmmAux, Xbyak::Reg64& rWorkRest, Xbyak::Reg64& rAux0, const Xbyak::Reg64& rAux1);
    void storeScalar(Xbyak::Reg64& rDst, Xbyak::Reg64& rToStoreCounter, Vmm& vmmSrc, Vmm& vAux);
    void shiftIdxAndGather(Vmm& vDst, Vmm& vAux, Vmask& mAux, bool isShortIdx, bool shiftFirst = true);
    void uni_vpgatherdd(Vmm& vDst, const Xbyak::Address& srcAddr, Vmask& vMask);
    void uni_vpcmpeqd(Vmask& vMask, Vmm& vOp0, Vmm& vOp2);

    const unsigned shufMask8bitUni[16]  = {0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080,
                                           0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080};
    const unsigned permMask8bitA2[8]    = {0, 4, 1, 5, 2, 6, 3, 7};
    const unsigned permMask8bitA5[16]   = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
    const unsigned* permMask8bitUni;

    const unsigned shufMask16bitUni[16] = {0x05040100, 0x0D0C0908, 0x80808080, 0x80808080, 0x05040100, 0x0D0C0908, 0x80808080, 0x80808080,
                                      0x05040100, 0x0D0C0908, 0x80808080, 0x80808080, 0x05040100, 0x0D0C0908, 0x80808080, 0x80808080};
    const unsigned permMask16bitA2[8]   = {0, 1, 4, 5, 2, 3, 6, 7};
    const unsigned permMask16bitA5[16]  = {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15};
    const unsigned* permMask16bitUni;

    const unsigned incVec[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
};

}  // namespace MKLDNNPlugin
