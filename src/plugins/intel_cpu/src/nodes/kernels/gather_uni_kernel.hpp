// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Gather kernel implements two approaches for indices calculation: "Short" and "Long".
// 1. Short approach is applicable for cases when the number of elements less or equal to vector register length.
// It just uses permutation of current indices vector to retrieve the next.
// 2. Long approach is applicable for cases when the number of elements is greater than vector register length.
// It increases indices in vector on vector length and normalizes upper bound of indices.
//
//                    SUPPORTED CASES
//--------------------------------------------------------------
//  After axis |         AVX512        |         AVX2          |
// (block) size| 32bit | 16bit |  8bit | 32bit | 16bit |  8bit |
//                      STATIC SHAPES
//      1      |   X   |   X   |   X   |   X   |   X   |   X   |
// >1 & <=vlen |   X   |   X   |   X   |   X   |       |       |
//                      DYNAMIC SHAPES
//      1      |   X   |   X   |   X   |   X   |   X   |   X   |
//--------------------------------------------------------------


#pragma once

#include "cpu/x64/jit_generator.hpp"
#include <dnnl_types.h>
#include "registers_pool.hpp"

namespace ov {
namespace intel_cpu {

struct jGatherConfParams {
    uint64_t dataTypeSize = 1lu;
    bool reverseIndexing = true;
    bool dynamicShapes = false;
    uint64_t batchDims = 0lu;
    uint64_t beforeAxisSize = 0lu;
    uint64_t specIdxSize = 0lu;
    uint64_t afterAxisSize = 0lu;
};

struct gatherJitExecArgs {
    const void* src;
    const void* indices;
    void* dst;
    const int* axisDim;
    const uint64_t* start;
    const uint64_t* specIndicesSize;
    const uint64_t* betweenBatchAndAxisSize;
    const uint64_t* axisAndAfterAxisSizeB;
    const uint64_t* srcAfterBatchSizeB;
    const int* permIdxMask;
    const int* beforeAxisDiff;

    const int* beforeAxisPermMask;
    const int* afterAxIdxB;
    const int* afterAxisPermMask;
    const uint64_t* afterAxisSize;
    const int* specIdxDiff;

    uint64_t workAmount = 0lu;
    uint64_t afterAxSize = 1lu;
    // Blocked short.
    uint64_t specIdxAndAfterAxIterB;
    uint64_t specIdxAndAfterAxSizeB;
    // Only static
    const int* specIdxB;
    const int* idxBatchSumB;
    const int* dataBeforeAxisSumB;
    uint64_t betweenBatchAndAxisIter;
};

struct jitGatherKernelInterface {
    virtual ~jitGatherKernelInterface() = default;
    virtual void initialize(const jGatherConfParams& jcp) = 0;
    virtual bool isSameParams(const jGatherConfParams& jcp) = 0;
    virtual void operator()(const gatherJitExecArgs *args) = 0;
    virtual void create_ker() = 0;
    virtual uint64_t getVecLen() const = 0;
    virtual uint64_t getDataElPerVec() const = 0;
    virtual uint64_t getIdxElPerVec() const = 0;
    static std::shared_ptr<jitGatherKernelInterface> createJitUniGatherKernel(x64::cpu_isa_t isa,
        uint64_t dataTypeSize, bool isDynamicNode, uint64_t afterAxisSize, uint64_t specIdxSize, uint64_t idxElPerVec);
};

using namespace dnnl::impl::cpu;

enum DataTypeSize {
    DataType32bit,
    DataType16bit,
    DataType8bit
};

enum Approach {
    Long,
    Short
};

enum AfterAxisCase {
    ElementwiseCase,
    BlockedCase
};

template <x64::cpu_isa_t isa>
class jitGatherKernelBase;

template<x64::cpu_isa_t isa>
using poolVmask = RegistersPool::Reg<typename jitGatherKernelBase<isa>::Vmask>;

template<x64::cpu_isa_t isa>
using poolVmm = RegistersPool::Reg<typename jitGatherKernelBase<isa>::Vmm>;

template <x64::cpu_isa_t isa>
class jitGatherKernelBase : public jitGatherKernelInterface, public x64::jit_generator {
public:
    using Vmm = typename x64::cpu_isa_traits<isa>::Vmm;
    using Vmask = typename dnnl::impl::utils::conditional<isa == x64::avx2, Xbyak::Ymm, Xbyak::Opmask>::type;
    jitGatherKernelBase(const char *name) : x64::jit_generator(name) {}

protected:
    static const uint32_t indicesTypeSize = sizeof(uint32_t);
    static const uint8_t idxTypeShift = 2;

    struct ShiftCalculator {
        virtual ~ShiftCalculator() = default;
        virtual std::tuple<poolVmask<isa> /*kDstMask*/, poolVmm<isa> /*vDstShifts*/> calcSrcShift(jitGatherKernelBase& kernel, bool shiftFirst) = 0;
    };

    template<AfterAxisCase C, Approach A, typename unused = void>
    struct ShiftCalculatorImpl : public ShiftCalculator {};

    template<typename unused>
    struct ShiftCalculatorImpl<ElementwiseCase, Short, unused> : public ShiftCalculator {
        void allocateRegisters(jitGatherKernelBase& kernel);
        void releaseRegisters();
        void uploadParamsForApproachSpecific(jitGatherKernelBase& kernel);
        std::tuple<poolVmask<isa> /*kDstMask*/, poolVmm<isa> /*vDstShifts*/> calcSrcShift(jitGatherKernelBase& kernel, bool shiftFirst) override;

        RegistersPool::Reg<Vmm> vmmBeforeAxDiffB;
        RegistersPool::Reg<Vmm> vmmSrcAfterBatchSizeB;
        RegistersPool::Reg<Vmm>  vmmPermIdxMask;
    };

    template<typename unused>
    struct ShiftCalculatorImpl<ElementwiseCase, Long, unused> : public ShiftCalculator {
        void allocateRegisters(jitGatherKernelBase& kernel);
        void releaseRegisters();
        void uploadParamsForApproachSpecific(jitGatherKernelBase& kernel);
        std::tuple<poolVmask<isa> /*kDstMask*/, poolVmm<isa> /*vDstShifts*/> calcSrcShift(jitGatherKernelBase& kernel, bool shiftFirst) override;

        Xbyak::Reg64 regBetweenBatchAndAxisSize;
        RegistersPool::Reg<Xbyak::Reg64> regSpecIdxSizeB;
        RegistersPool::Reg<Xbyak::Reg64> regBetweenBatchAndAxisIter;
        RegistersPool::Reg<Xbyak::Reg64> regIdxIter;

        RegistersPool::Reg<Vmm> vmmIdxBatchSumB;
        RegistersPool::Reg<Vmm> vmmVecLenB;
        RegistersPool::Reg<Vmm> vmmAxisAndAfterAxisSizeB;
    };

    template<typename unused>
    struct ShiftCalculatorImpl<BlockedCase, Short, unused> : public ShiftCalculator {
        void allocateRegisters(jitGatherKernelBase& kernel);
        void uploadParamsForApproachSpecific(jitGatherKernelBase& kernel);
        std::tuple<poolVmask<isa> /*kDstMask*/, poolVmm<isa> /*vDstShifts*/> calcSrcShift(jitGatherKernelBase& kernel, bool shiftFirst) override;

        RegistersPool::Reg<Xbyak::Reg64> rSpecIdxAndAfterAxIterB;
        RegistersPool::Reg<Xbyak::Reg64> rSpecIdxAndAfterAxSizeB;

        RegistersPool::Reg<Vmm> vmmAxisAndAfterAxisSizeB;
        RegistersPool::Reg<Vmm> vmmSrcAfterBatchSizeB;
        RegistersPool::Reg<Vmm> vmmAfterAxisIdxB;
        RegistersPool::Reg<Vmm> vmmAfterAxisPermMask;
        RegistersPool::Reg<Vmm> vmmSpecIdxDiff;
        RegistersPool::Reg<Vmm> vmmAfterAxisSize;
        RegistersPool::Reg<Vmm> vmmBeforeAxPermMask;
    };


public:
    void initialize(const jGatherConfParams& jcp) override;
    bool isSameParams(const jGatherConfParams& jcp) override;
    void create_ker() override;
    void generate() override;
    void operator() (const gatherJitExecArgs *args) override {
        assert(ker_);
        ker_(args);
    }
    uint64_t getVecLen() const override { return x64::cpu_isa_traits<isa>::vlen; }
    uint64_t getDataElPerVec() const override { return getVecLen() / getDataTypeSize(); }
    uint64_t getIdxElPerVec() const override { return getVecLen() / indicesTypeSize; }
    virtual uint64_t getDataTypeSize() const = 0;
    virtual uint8_t getDataTypeShift() const = 0;

protected:
    std::tuple<poolVmask<isa> /*kDstMask*/, poolVmm<isa> /*vDstShifts*/> calculateIndexesForShortCase(
            Vmm& srcBeforeAxisSumB, Vmm& srcAfterBatchSizeB);
    poolVmask<isa> calcAllOnesMask(Vmm& vAux);
    void uploadParamPtrWithVpbroadcastd(const Vmm& vmmDest, size_t offset);
    void uploadParamPtrWithVmovups(const Vmm& vmmDest, size_t offset);
    virtual void generateDynamicitySpecific() = 0;
    void process(ShiftCalculator& shiftCalculator);
    virtual void processDataTypeSpecific(ShiftCalculator& shiftCalculator) = 0;
    void tail(ShiftCalculator& shiftCalculator, bool shiftFirst);
    poolVmm<isa> shiftIdxAndGather(ShiftCalculator& shiftCalculator, bool shiftFirst);
    void uniVpGatherDd(Vmm& vDst, const Xbyak::Address& srcAddr, Vmask& vMask);
    poolVmask<isa> normalizeIndexesAndCalcShifts(Vmm& rawIndices, poolVmask<isa> kDstMask = poolVmask<isa>());
    void fillRestWorkMask(Vmask& kMask, const Xbyak::Reg& rWorkRest);
    void combineMasks(Vmask& maskA, const Vmask& maskB);
    void normWithUpperBound(Vmm& vTarget, Vmm& vMax, Vmask& kAuxMask);
    void fillVlenVector(RegistersPool::Reg<Vmm>& vmmVecLenB);
    void storeVectorPart(const Xbyak::Reg& rDst, const Xbyak::Reg& rToStoreCounter, Vmm& vmmSrc);

protected:
    void (*ker_)(const gatherJitExecArgs *);
    using Reg64 = Xbyak::Reg64;
    using Operand = Xbyak::Operand;
    RegistersPool::Ptr regPool = RegistersPool::create<isa>({
        // the list of the registers to be excluded from pool
        Reg64(Operand::RAX), Reg64(Operand::RCX), Reg64(Operand::RDX), Reg64(Operand::RBX),
        Reg64(Operand::RBP), Reg64(Operand::RDI),
        Xbyak::Opmask(0), // Do not use k0 with gather instruction. The k0 has special meaning there.
    });
    uint64_t beforeAxisSize = 0lu;
    uint64_t specIdxSize = 0lu;
    uint64_t batchDims = 0lu;
    uint64_t afterAxisSize = 0lu;
    bool reverseIndexing = true;
    bool dynamicShapes = false;

    const Xbyak::Reg64 regParams = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);
    const RegistersPool::Reg<Xbyak::Reg64> regDst {regPool, 9};
    const RegistersPool::Reg<Xbyak::Reg64> regIndices {regPool, 10};
    const RegistersPool::Reg<Xbyak::Reg64> regWorkAmount {regPool, 12};
    RegistersPool::Reg<Vmm> vmmSpecIdxSizeB {this->regPool, 10};
    RegistersPool::Reg<Vmm> vmmSpecIdxB {this->regPool, 9};
    RegistersPool::Reg<Vmm> vmmSrcBeforeAxisSumB {this->regPool, 8};

private:
    const RegistersPool::Reg<Xbyak::Reg64> regSrc {regPool, 8};
    RegistersPool::Reg<Vmm> vmmZeros {regPool, 7};
    RegistersPool::Reg<Vmm> vmmAxisDim {regPool, 11};
};

template<x64::cpu_isa_t isa, DataTypeSize S>
struct jitGatherKernelForDataTypeSize : public jitGatherKernelBase<isa> {};

template<x64::cpu_isa_t isa>
struct jitGatherKernelForDataTypeSize<isa, DataType32bit> : public jitGatherKernelBase<isa> {
    using Vmm = typename jitGatherKernelBase<isa>::Vmm;
    jitGatherKernelForDataTypeSize(const char *name) : jitGatherKernelBase<isa>(name) {}
    uint64_t getDataTypeSize() const override { return 4lu; }
    uint8_t getDataTypeShift() const override { return 2; }
    void processDataTypeSpecific(typename jitGatherKernelBase<isa>::ShiftCalculator& shiftCalculator) override;
};

template<x64::cpu_isa_t isa>
struct jitGatherKernelForDataTypeSize<isa, DataType16bit> : public jitGatherKernelBase<isa> {
    using Vmm = typename jitGatherKernelBase<isa>::Vmm;
    jitGatherKernelForDataTypeSize(const char *name) : jitGatherKernelBase<isa>(name) {}
    uint64_t getDataTypeSize() const override { return 2lu; }
    uint8_t getDataTypeShift() const override { return 1; }
    void processDataTypeSpecific(typename jitGatherKernelBase<isa>::ShiftCalculator& shiftCalculator) override;
};


template<x64::cpu_isa_t isa>
struct jitGatherKernelForDataTypeSize<isa, DataType8bit> : public jitGatherKernelBase<isa> {
    using Vmm = typename jitGatherKernelBase<isa>::Vmm;
    jitGatherKernelForDataTypeSize(const char *name) : jitGatherKernelBase<isa>(name) {}
    uint64_t getDataTypeSize() const override { return 1lu; }
    uint8_t getDataTypeShift() const override { return 0; }
    void processDataTypeSpecific(typename jitGatherKernelBase<isa>::ShiftCalculator& shiftCalculator) override;
    poolVmm<isa> calculateIdxShiftsForHalfIteration(typename jitGatherKernelBase<isa>::ShiftCalculator& shiftCalculator,
                                                    Vmm& vShufMask, bool shiftFirst, poolVmm<isa> halfPart = poolVmm<isa>());
};


template<x64::cpu_isa_t isa, DataTypeSize S, AfterAxisCase C, Approach A>
struct jitGatherKernelForStaticShapes : public jitGatherKernelForDataTypeSize<isa, S> {
    using Vmm = typename jitGatherKernelBase<isa>::Vmm;
    jitGatherKernelForStaticShapes() : jitGatherKernelForDataTypeSize<isa, S>(jit_name()) {}
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jitGatherKernelForDynamicity)
protected:
    void generateDynamicitySpecific() override;

    typename jitGatherKernelBase<isa>::template ShiftCalculatorImpl<C, A> shiftCalculator;
};


template<x64::cpu_isa_t isa, DataTypeSize S>
struct jitGatherKernelForDynamicShapes : public jitGatherKernelForDataTypeSize<isa, S> {
    using Vmm = typename jitGatherKernelBase<isa>::Vmm;
    jitGatherKernelForDynamicShapes() : jitGatherKernelForDataTypeSize<isa, S>(jit_name()) {}
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jitGatherKernelForDynamicShapes)
protected:
    void generateDynamicitySpecific() override;

    uint64_t idxElPerVec = 0lu;
};

}   // namespace intel_cpu
}   // namespace ov
