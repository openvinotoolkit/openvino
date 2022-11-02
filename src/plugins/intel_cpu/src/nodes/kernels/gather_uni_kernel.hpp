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

#include <utility>
#include "registers_pool.hpp"
#include "stack_allocator.hpp"

namespace ov {
namespace intel_cpu {

template<typename TReg>
struct RegisterValue {
    using ValueInitializer = std::function<void(TReg& valueToInitialize)>;
    static constexpr int anyIdx = -1;

    RegisterValue() {}

    RegisterValue(ValueInitializer initializer)
            : valueInitializer(std::move(initializer)) {
    }

    operator TReg&() { ensureValid(); return reg; }
    operator const TReg&() const { ensureValid(); return reg; }
    int getIdx() { ensureValid(); return reg.getIdx(); }
    bool isValueInRegister() { return reg.isInitialized(); }
    bool isValueInStack() { return static_cast<bool>(stackedReg); }
    bool isInitialized() { return this->isValueInRegister() || isValueInStack(); }

    void initialize(RegistersPool::Ptr& pool, int requestedIdx = anyIdx) {
        if (isInitialized()) {
            IE_THROW() << "Can not initialize RegisterValue, already initialized";
        }
        reg = RegistersPool::Reg<TReg>{pool, requestedIdx};
        valueInitializer(static_cast<TReg&>(reg));
    }
    int reset() {
        if (!isInitialized()) {
            IE_THROW() << "Can not reset RegisterValue, it is not initialized";
        }
        int idx = this->reg.getIdx();
        this->reg.release();
        stackedReg.reset();
        return idx;
    }
    int saveToStack(std::shared_ptr<StackAllocator>& allocator) {
        if (!this->isValueInRegister()) {
            IE_THROW() << "The RegisterValue is ether not initialized or already saved to stack in RegisterValue::saveToStack()";
        }
        stackedReg = std::make_shared<StackAllocator::Reg<TReg>>(*allocator);
        stack_mov(*stackedReg, this->reg);
        int idx = this->reg.getIdx();
        this->reg.release();
        return idx;
    }
    void loadFromStack(RegistersPool::Ptr& pool, int requestedIdx = anyIdx) {
        if (!isValueInStack()) {
            IE_THROW() << "Inconsistency in RegisterValue::loadFromStack()";
        }
        this->reg = RegistersPool::Reg<TReg>{pool, requestedIdx};
        stack_mov(this->reg, *stackedReg);
        stackedReg.reset();
    }

    RegisterValue& operator=(RegisterValue&& other)  noexcept {
        stackedReg = std::move(other.stackedReg);
        reg = std::move(other.reg);
        valueInitializer = std::move(other.valueInitializer);
        return *this;
    }

    RegisterValue(RegisterValue&& other)  noexcept
            : stackedReg(std::move(other.stackedReg))
            , reg(std::move(other.reg))
            , valueInitializer(std::move(other.valueInitializer)) {}

private:
    void ensureValid() {
        if (!isValueInRegister()) {
            IE_THROW() << "Failed to use the RegisterValue, the value is not in the register";
        }
    }

private:
    ValueInitializer valueInitializer;
    std::shared_ptr<StackAllocator::Reg<TReg>> stackedReg;
    RegistersPool::Reg<TReg> reg;
};


struct jGatherConfParams {
    uint64_t dataTypeSize = 1lu;
    bool reverseIndexing = true;
    bool dynamicShapes = false;
    uint64_t batchDims = 0lu;
    uint64_t beforeAxisSize = 0lu;
    uint64_t specIdxSize = 0lu;
    uint64_t afterAxisSize = 0lu;
    uint64_t simdVecSize = 16lu;
    uint64_t idxElPerVec = 1lu;
    uint64_t dataElPerVec = 1lu;
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
    // Blocked short dynamic
    uint64_t specIdxAndAfterAxSize;
    uint64_t beforeAxisSize;
    uint64_t specIdxAndAfterAxisSizeIsPowerOf2;
    uint16_t afterAxisSizeIsPowerOf2;
    uint64_t specIdxSize;
    // Only static
    const int* specIdxB;
    const int* idxBatchSumB;
    const int* dataBeforeAxisSumB;
    uint64_t betweenBatchAndAxisIter;
};

struct GatherShapeParameters {
    int axisDim = 0;
    uint64_t betweenBatchAndAxisSize = 0lu;
    uint64_t specIdxAndAfterAxSizeB = 0lu;
    uint64_t beforeBatchSize = 0lu;
    uint64_t specIndicesSize = 0lu;
    uint64_t afterAxisSizeInBytes = 0lu;
    uint64_t axisAndAfterAxisSizeInBytes = 0lu;
    uint64_t srcAfterBatchSizeInBytes = 0lu;
    uint64_t beforeAxisSize = 0lu;
    uint64_t afterAxisSize = 0lu;
    uint64_t totalWork = 0lu;
    uint64_t dataTypeSize = 1lu;
    uint64_t simdVecSize = 16lu;
    uint64_t idxElPerVec = 1lu;
    uint64_t dataElPerVec = 1lu;
    static constexpr uint64_t idxTypeSize = sizeof(int);

    struct PerThread {
        std::vector<int> specIdxInBytes;
        std::vector<int> permIdxMask;
        std::vector<int> srcBeforeAxisDiff;
        std::vector<int> idxBatchSumInBytes;
        std::vector<int> dataBeforeAxisSumInBytes;

        std::vector<int> afterAxIdxInBytes;
        std::vector<int> specIdxDiff;
        std::vector<int> beforeAxPermMask;
        std::vector<int> afterAxPermMask;
        int betweenBatchAndAxisIter = 0;
        int specIdxAndAfterAxIterB = 0;

        uint64_t workAmount = 0;
        uint64_t dstStart = 0;
    };

    GatherShapeParameters() = default;
    void initStatic(bool isAxisInputConst, bool isDataShapeStat, bool isIdxShapeStat,
                             const VectorDims& dataDims, const VectorDims& idxDims, int axis, int batchDims) {
        if (isAxisInputConst && isDataShapeStat) {
            axisDim = dataDims[axis];
            beforeAxisSize = std::accumulate(dataDims.begin(), dataDims.begin() + axis, 1lu, std::multiplies<Dim>());
            betweenBatchAndAxisSize = std::accumulate(dataDims.begin() + batchDims, dataDims.begin() + axis, 1lu, std::multiplies<Dim>());
            afterAxisSize = std::accumulate(dataDims.begin() + axis + 1, dataDims.end(), 1lu, std::multiplies<Dim>());

            afterAxisSizeInBytes = afterAxisSize * dataTypeSize;
            axisAndAfterAxisSizeInBytes = axisDim * afterAxisSizeInBytes;
            srcAfterBatchSizeInBytes = betweenBatchAndAxisSize * axisAndAfterAxisSizeInBytes;
        }
        if (isDataShapeStat) {
            beforeBatchSize = std::accumulate(dataDims.begin(), dataDims.begin() + batchDims, 1lu, std::multiplies<Dim>());
        }
        if (isIdxShapeStat) {
            specIndicesSize = std::accumulate(idxDims.begin() + batchDims, idxDims.end(), 1lu, std::multiplies<Dim>());

            if (isDataShapeStat) {
                specIdxAndAfterAxSizeB = specIndicesSize * afterAxisSizeInBytes;
                totalWork = beforeBatchSize * betweenBatchAndAxisSize * specIndicesSize * afterAxisSize;
            }
        }
    }

    void initDynamic(bool isAxisInputConst, bool isDataShapeStat, bool isIdxShapeStat,
                             const VectorDims& dataDims, const VectorDims& idxDims, int axis, int batchDims) {
        if (!isDataShapeStat || !isAxisInputConst) {
            axisDim = dataDims[axis];
            beforeAxisSize = std::accumulate(dataDims.begin(), dataDims.begin() + axis, 1lu, std::multiplies<Dim>());
            beforeBatchSize = std::accumulate(dataDims.begin(), dataDims.begin() + batchDims, 1lu, std::multiplies<uint64_t>());
            betweenBatchAndAxisSize = std::accumulate(dataDims.begin() + batchDims, dataDims.begin() + axis, 1lu, std::multiplies<uint64_t>());
            afterAxisSize = std::accumulate(dataDims.begin() + axis + 1, dataDims.end(), 1lu, std::multiplies<uint64_t>());

            afterAxisSizeInBytes = afterAxisSize * dataTypeSize;
            axisAndAfterAxisSizeInBytes = axisDim * afterAxisSizeInBytes;
            srcAfterBatchSizeInBytes = betweenBatchAndAxisSize * axisAndAfterAxisSizeInBytes;

            if (isIdxShapeStat) {
                specIdxAndAfterAxSizeB = specIndicesSize * afterAxisSizeInBytes;
                totalWork = beforeBatchSize * betweenBatchAndAxisSize * specIndicesSize * afterAxisSize;
            }
        }

        if (!isIdxShapeStat) {
            specIndicesSize = std::accumulate(idxDims.begin() + batchDims, idxDims.end(), 1lu, std::multiplies<uint64_t>());

            specIdxAndAfterAxSizeB = specIndicesSize * afterAxisSizeInBytes;
            totalWork = beforeBatchSize * betweenBatchAndAxisSize * specIndicesSize * afterAxisSize;
        }
    }

    void fillPerThread(PerThread& p, uint64_t dstStart, uint64_t dstEnd) {
        p.workAmount = dstEnd - dstStart;
        p.dstStart = dstStart;
        p.specIdxInBytes.resize(dataElPerVec);
        p.idxBatchSumInBytes.resize(dataElPerVec);
        p.dataBeforeAxisSumInBytes.resize(dataElPerVec);
        p.betweenBatchAndAxisIter = (dstStart / specIndicesSize) % betweenBatchAndAxisSize;
        for (uint64_t j = 0lu; j < dataElPerVec; j++) {
            p.specIdxInBytes[j] = (((dstStart + j) / afterAxisSize) % specIndicesSize) * idxTypeSize;
            p.idxBatchSumInBytes[j] = ((dstStart + j) / (betweenBatchAndAxisSize * specIndicesSize * afterAxisSize)) *
                                      specIndicesSize * idxTypeSize;
            p.dataBeforeAxisSumInBytes[j] = ((dstStart + j) / (specIndicesSize * afterAxisSize)) *
                                            axisAndAfterAxisSizeInBytes;
        }
        initShortParams(p, dstStart);
    }

    void initShortParams(PerThread& p, const uint64_t start) {
        fillBeforeAxisDiff(p.srcBeforeAxisDiff, start);
        if (afterAxisSize == 1) { // Elementwise gather.
            if (specIndicesSize >= idxElPerVec)
                return; // Is not a short case.

            fillPermIdxMask(p.permIdxMask);
        } else { // Blocked gather.
            if (afterAxisSize > idxElPerVec)
                return; // Is not a short case.

            p.afterAxIdxInBytes.resize(idxElPerVec);
            p.afterAxPermMask.resize(idxElPerVec);
            p.beforeAxPermMask.resize(idxElPerVec);
            p.specIdxDiff.resize(idxElPerVec);

            int secondStart = start + idxElPerVec;
            for (int i = 0; i < idxElPerVec; i++) {
                p.afterAxIdxInBytes[i] = (start + i) % afterAxisSize;
                p.specIdxDiff[i] = (((secondStart + i) / afterAxisSize) % specIndicesSize) * idxTypeSize - p.specIdxInBytes[i];
                if (p.specIdxDiff[i] < 0)
                    p.specIdxDiff[i] += specIndicesSize * idxTypeSize;

                p.afterAxIdxInBytes[i] *= dataTypeSize;
                p.afterAxPermMask[i] = idxElPerVec - afterAxisSize + i;
                for (size_t j = 0lu; j < 6lu; j++) {
                    if (p.afterAxPermMask[i] >= idxElPerVec)
                        p.afterAxPermMask[i] -= afterAxisSize;
                }
            }
            if (specIndicesSize * afterAxisSize < idxElPerVec) {
                p.beforeAxPermMask[0] = idxElPerVec - specIndicesSize * afterAxisSize;
                for (int i = 1; i < idxElPerVec; i++) {
                    p.beforeAxPermMask[i] = p.beforeAxPermMask[i - 1] + 1;
                    if (p.beforeAxPermMask[i] == idxElPerVec)
                        p.beforeAxPermMask[i] = idxElPerVec - specIndicesSize * afterAxisSize;
                }
            }

            p.specIdxAndAfterAxIterB = (start * dataTypeSize) % specIdxAndAfterAxSizeB;
        }
    }

    void fillPermIdxMask(std::vector<int>& permIdxMask) {
        permIdxMask.resize(idxElPerVec);
        permIdxMask[0] = idxElPerVec - specIndicesSize;
        for (int i = 1; i < idxElPerVec; i++) {
            permIdxMask[i] = permIdxMask[i - 1] + 1;
            if (permIdxMask[i] == idxElPerVec)
                permIdxMask[i] = idxElPerVec - specIndicesSize;
        }
    }

    void fillBeforeAxisDiff(std::vector<int>& srcBeforeAxisDiff, uint64_t start) {
        if (afterAxisSize == 1) { // Elementwise gather.
            srcBeforeAxisDiff.resize(idxElPerVec);
            const int div = idxElPerVec / specIndicesSize;
            const int remainder = idxElPerVec % specIndicesSize;
            for (uint64_t i = 0; i < idxElPerVec; i++) {
                if (((start + i) % specIndicesSize) < (specIndicesSize - remainder)) {
                    srcBeforeAxisDiff[i] = axisDim * div;
                } else {
                    srcBeforeAxisDiff[i] = axisDim * (div + 1);
                }
            }
        } else {
            srcBeforeAxisDiff.resize(idxElPerVec);
            for (int i = 0; i < idxElPerVec; i++) {
                srcBeforeAxisDiff[i] = ((start + i + idxElPerVec) / (specIndicesSize * afterAxisSize)) * axisAndAfterAxisSizeInBytes -
                                         ((start + i) / (specIndicesSize * afterAxisSize)) * axisAndAfterAxisSizeInBytes;
            }
        }
    }

    gatherJitExecArgs createArgStatic(uint8_t* dstData, const void* srcData, const void* srcIndices, const PerThread& p) {
        auto arg = gatherJitExecArgs();

        arg.src = srcData;
        arg.dst = dstData + p.dstStart * dataTypeSize;
        arg.indices = srcIndices;
        arg.start = &p.dstStart;
        arg.axisDim = &axisDim;
        arg.afterAxSize = afterAxisSize;
        arg.axisAndAfterAxisSizeB = &axisAndAfterAxisSizeInBytes;
        arg.srcAfterBatchSizeB = &srcAfterBatchSizeInBytes;
        arg.betweenBatchAndAxisSize = &betweenBatchAndAxisSize;
        arg.specIndicesSize = &specIndicesSize;
        arg.workAmount = p.workAmount;
        arg.specIdxB = p.specIdxInBytes.data();
        arg.idxBatchSumB = p.idxBatchSumInBytes.data();
        arg.dataBeforeAxisSumB = p.dataBeforeAxisSumInBytes.data();
        arg.betweenBatchAndAxisIter = p.betweenBatchAndAxisIter;

        if (afterAxisSize == 1 && specIndicesSize < idxElPerVec) { // Elementwise short case.
            arg.permIdxMask = p.permIdxMask.data();
            arg.beforeAxisDiff = p.srcBeforeAxisDiff.data();
        } else if (afterAxisSize > 1 && afterAxisSize <= dataElPerVec) { // Blocked short case.
            arg.afterAxIdxB = p.afterAxIdxInBytes.data();
            arg.specIdxDiff = p.specIdxDiff.data();
            arg.beforeAxisDiff = p.srcBeforeAxisDiff.data();
            arg.beforeAxisPermMask = p.beforeAxPermMask.data();
            arg.afterAxisPermMask = p.afterAxPermMask.data();
            arg.afterAxisSize = &afterAxisSize;
            arg.specIdxAndAfterAxIterB = p.specIdxAndAfterAxIterB;
            arg.specIdxAndAfterAxSizeB = specIdxAndAfterAxSizeB;
        }
        return arg;
    }

    gatherJitExecArgs createArgDynamic(uint8_t* dstData, const void* srcData, const void* srcIndices, const PerThread& p) {
        auto arg = gatherJitExecArgs();

        arg.src = srcData;
        arg.dst = dstData + p.dstStart * dataTypeSize;
        arg.indices = srcIndices;
        arg.start = &p.dstStart;
        arg.axisDim = &axisDim;
        arg.afterAxSize = afterAxisSize;
        arg.axisAndAfterAxisSizeB = &axisAndAfterAxisSizeInBytes;
        arg.srcAfterBatchSizeB = &srcAfterBatchSizeInBytes;
        arg.betweenBatchAndAxisSize = &betweenBatchAndAxisSize;
        arg.specIndicesSize = &specIndicesSize;
        arg.workAmount = p.workAmount;
        arg.specIdxB = p.specIdxInBytes.data();
        arg.idxBatchSumB = p.idxBatchSumInBytes.data();
        arg.dataBeforeAxisSumB = p.dataBeforeAxisSumInBytes.data();
        arg.betweenBatchAndAxisIter = p.betweenBatchAndAxisIter;

        if (afterAxisSize == 1 && specIndicesSize < idxElPerVec) { // Elementwise short case.
            arg.permIdxMask = p.permIdxMask.data();
            arg.beforeAxisDiff = p.srcBeforeAxisDiff.data();
        } else if (afterAxisSize > 1 && afterAxisSize <= dataElPerVec) { // Blocked short case.
            arg.afterAxIdxB = p.afterAxIdxInBytes.data();
            arg.specIdxDiff = p.specIdxDiff.data();
            arg.beforeAxisDiff = p.srcBeforeAxisDiff.data();
            arg.beforeAxisPermMask = p.beforeAxPermMask.data();
            arg.afterAxisPermMask = p.afterAxPermMask.data();
            arg.afterAxisSize = &afterAxisSize;
            arg.specIdxAndAfterAxIterB = p.specIdxAndAfterAxIterB;
            arg.specIdxAndAfterAxSizeB = specIdxAndAfterAxSizeB;
            arg.specIdxAndAfterAxSize = specIndicesSize * arg.afterAxSize;
            arg.beforeAxisSize = beforeAxisSize;
            arg.specIdxSize = specIndicesSize;
            arg.specIdxAndAfterAxisSizeIsPowerOf2 = arg.specIdxAndAfterAxSize == 1 || arg.specIdxAndAfterAxSize == 2 ||
                 arg.specIdxAndAfterAxSize == 4 || arg.specIdxAndAfterAxSize == 8 || arg.specIdxAndAfterAxSize == 16 ? 1 : 0;
            arg.afterAxisSizeIsPowerOf2 = afterAxisSize == 1 || afterAxisSize == 2 || afterAxisSize == 4 ||
                    afterAxisSize == 8 || afterAxisSize == 16 ? 1 : 0;
        }
        return arg;
    }
};

struct jitGatherKernelInterface {
    virtual ~jitGatherKernelInterface() = default;
    virtual void initialize(const jGatherConfParams& jcp) = 0;
    virtual bool isSameParams(const jGatherConfParams& jcp) = 0;
    virtual void operator()(const gatherJitExecArgs *args) = 0;
    virtual void create_ker() = 0;
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
        RegistersPool::Reg<Xbyak::Reg64> rSpecIdxAndAfterAxSize;
        RegistersPool::Reg<Xbyak::Reg64> rBeforeAxisSize;
        RegistersPool::Reg<Xbyak::Reg64> rSpecIdxAndAfterAxisSizeIsPowerOf2;
        RegistersPool::Reg<Xbyak::Reg64> rAfterAxisSizeIsPowerOf2;
        Xbyak::Reg64 rSpecIdxSize{Xbyak::Operand::RCX};

        RegisterValue<Vmm> vmmAxisAndAfterAxisSizeB;
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
    poolVmask<isa> fillRestWorkMask(const Xbyak::Reg& rWorkRest);
    void combineMasks(Vmask& maskA, const Vmask& maskB);
    void normWithUpperBound(Vmm& vTarget, Vmm& vMax);
    void fillVlenVector(RegistersPool::Reg<Vmm>& vmmVecLenB);
    void storeVectorPart(const Xbyak::Reg& rDst, const Xbyak::Reg& rToStoreCounter, Vmm& vmmData);
    void generateForDynamicShapes();
    void generateForStaticShapes();

protected:
    void (*ker_)(const gatherJitExecArgs *);
    using Reg64 = Xbyak::Reg64;
    using Operand = Xbyak::Operand;
    RegistersPool::Ptr regPool = RegistersPool::create<isa>({
        // the list of the registers to be excluded from pool
        Reg64(Operand::RCX), Reg64(Operand::RBP), Reg64(Operand::RDI),
        Xbyak::Opmask(0), // Do not use k0 with gather instruction. The k0 has special meaning there.
    });
    std::shared_ptr<StackAllocator> stackAllocator;
    uint64_t simdVecSize = 16lu;
    uint64_t idxElPerVec = 1lu;
    uint64_t dataElPerVec = 1lu;
    uint64_t beforeAxisSize = 0lu;
    uint64_t specIdxSize = 0lu;
    uint64_t batchDims = 0lu;
    uint64_t afterAxisSize = 0lu;
    bool reverseIndexing = true;
    bool dynamicShapes = false;
    bool isLessSimdRegistersCase = false;

    const Xbyak::Reg64 regParams = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);
    const RegistersPool::Reg<Xbyak::Reg64> regDst {regPool, 9};
    const RegistersPool::Reg<Xbyak::Reg64> regIndices {regPool, 10};
    const RegistersPool::Reg<Xbyak::Reg64> regWorkAmount {regPool, 12};
    RegistersPool::Reg<Vmm> vmmSpecIdxSizeB {this->regPool, 10};
    RegistersPool::Reg<Vmm> vmmSpecIdxB {this->regPool, 9};
    RegisterValue<Vmm> vmmSrcBeforeAxisSumB;
    static const int vmmAxisAndAfterAxisSizeBIndx {12};
    static const int vmmZerosIdx {7};
    RegisterValue<Vmm> vmmZeros;

private:
    const RegistersPool::Reg<Xbyak::Reg64> regSrc {regPool, 8};
    const int vmmAxisDimIdx {11};
    RegisterValue<Vmm> vmmAxisDim;
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
