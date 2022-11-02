// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Gather kernel implements two cases depending on shapes: "Blocked" and "Elementwise".
// And each of these cases have subcases: "Short" and "Long".
// The Elementwise approach is applicable for cases when there is only one data element per one index element,
// otherwise it will be Blocked approach.
// The Elementwise/Short case is when the number of indices in one batch is less or equal to vector register length,
// in other case it will be Elementwise/Long case.
// The Blocked/Short case is when the number of data elements per one index element is less or equal to vector register length,
// in other case it will be Blocked/Long case.

// The implementation map for the JIT kernel for the Gather operation.
//- avx512
//    - dynamic shapes
//        - Elementwise case
//            - Short subcase  Implemented
//            - Long subcase   Implemented
//        - Blocked case
//            - Short subcase  Not implemented
//            - Long subcase   Not implemented
//    - static shapes
//        - Elementwise case
//            - Short subcase  Implemented
//            - Long subcase   Implemented
//        - Blocked case
//            - Short subcase  Implemented
//            - Long subcase   Not implemented
//- avx2
//    - dynamic shapes
//        - Elementwise case
//            - Short subcase  Implemented
//            - Long subcase   Implemented
//        - Blocked case
//            - Short subcase  Not implemented
//            - Long subcase   Not implemented
//    - static shapes
//        - Elementwise case
//            - Short subcase  Implemented
//            - Long subcase   Implemented
//        - Blocked case
//            - Short subcase  Implemented only for 32 bit data type
//            - Long subcase   Not implemented
//- SSE4.1                     Not implemented

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
        virtual void allocateRegisters(jitGatherKernelBase& kernel) = 0;
        virtual void uploadParamsForApproachSpecific(jitGatherKernelBase& kernel) = 0;
        virtual std::tuple<poolVmask<isa> /*kDstMask*/, poolVmm<isa> /*vDstShifts*/> calcSrcShift(jitGatherKernelBase& kernel, bool shiftFirst) = 0;
    };

    template<AfterAxisCase C, Approach A, typename unused = void>
    struct ShiftCalculatorImpl : public ShiftCalculator {};

    template<typename unused>
    struct ShiftCalculatorImpl<ElementwiseCase, Short, unused> : public ShiftCalculator {
        void allocateRegisters(jitGatherKernelBase& kernel) override;
        void releaseRegisters();
        void uploadParamsForApproachSpecific(jitGatherKernelBase& kernel) override;
        std::tuple<poolVmask<isa> /*kDstMask*/, poolVmm<isa> /*vDstShifts*/> calcSrcShift(jitGatherKernelBase& kernel, bool shiftFirst) override;

        RegistersPool::Reg<Vmm> vmmBeforeAxDiffB;
        RegistersPool::Reg<Vmm> vmmSrcAfterBatchSizeB;
        RegistersPool::Reg<Vmm>  vmmPermIdxMask;
    };

    template<typename unused>
    struct ShiftCalculatorImpl<ElementwiseCase, Long, unused> : public ShiftCalculator {
        void allocateRegisters(jitGatherKernelBase& kernel) override;
        void releaseRegisters();
        void uploadParamsForApproachSpecific(jitGatherKernelBase& kernel) override;
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
        void allocateRegisters(jitGatherKernelBase& kernel) override;
        void uploadParamsForApproachSpecific(jitGatherKernelBase& kernel) override;
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
    virtual ShiftCalculator& getShiftCalculator() { IE_THROW() << "Inconsistency in jitGatherKernelBase::getShiftCalculator()"; }
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
    void generateForDynamicShapes();
    void generateForStaticShapes();

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
    uint64_t idxElPerVec = 0lu;
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
    jitGatherKernelForStaticShapes()
            : jitGatherKernelForDataTypeSize<isa, S>(jit_name())
            , shiftCalculator(std::make_shared<typename jitGatherKernelBase<isa>::template ShiftCalculatorImpl<C, A>>())
    {}
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jitGatherKernelForDynamicity)
protected:
    typename jitGatherKernelBase<isa>::ShiftCalculator& getShiftCalculator() override { return *shiftCalculator; }

    std::shared_ptr<typename jitGatherKernelBase<isa>::ShiftCalculator> shiftCalculator;
};


template<x64::cpu_isa_t isa, DataTypeSize S>
struct jitGatherKernelForDynamicShapes : public jitGatherKernelForDataTypeSize<isa, S> {
    using Vmm = typename jitGatherKernelBase<isa>::Vmm;
    jitGatherKernelForDynamicShapes() : jitGatherKernelForDataTypeSize<isa, S>(jit_name()) {}
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jitGatherKernelForDynamicShapes)
};

}   // namespace intel_cpu
}   // namespace ov
