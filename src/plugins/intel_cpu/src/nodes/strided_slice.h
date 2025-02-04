// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class StridedSlice : public Node {
public:
    StridedSlice(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    bool neverExecute() const override;
    bool isExecutable() const override;
    bool needShapeInfer() const override;

    struct StridedSliceAttributes {
        std::vector<int> begin;
        std::vector<int> end;
        std::vector<int> stride;
        std::vector<int> axes;

        std::vector<int> beginMask;
        std::vector<int> endMask;
        std::vector<int> ellipsisMask;
        std::vector<int> newAxisMask;
        std::vector<int> shrinkAxisMask;

        VectorDims beginDims;
        VectorDims endDims;
        VectorDims strideDims;
        VectorDims axesDims;

        bool equalDims = false;
        size_t dataSize = 1lu;
        int ellipsisMaskCounter = 0;
        bool isStridedSliceOp = true;
        bool isSliceScatterOp = false;
        int ellipsisPos1 = -1;
        bool hasConstInputs = false;
        size_t DATA_ID = 0;
        size_t BEGIN_ID = 1;
        size_t END_ID = 2;
        size_t STRIDE_ID = 3;
        size_t AXES_ID = 4;
        size_t UPDATES_ID = 1;
    } attrs;

protected:
    bool needPrepareParams() const override;
    void prepareParams() override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

private:
    class StridedSliceExecutor {
    public:
        StridedSliceExecutor(const StridedSliceAttributes& attrs,
                             const std::vector<MemoryCPtr>& srcMemory,
                             const std::vector<MemoryCPtr>& dstMemory) {}
        virtual void exec(const std::vector<MemoryCPtr>& srcMemory, const std::vector<MemoryCPtr>& dstMemory) = 0;
        virtual ~StridedSliceExecutor() = default;
    };

    class StridedSliceCommonExecutor : public StridedSliceExecutor {
    public:
        StridedSliceCommonExecutor(const StridedSliceAttributes& attrs,
                                   const std::vector<MemoryCPtr>& srcMemory,
                                   const std::vector<MemoryCPtr>& dstMemory);
        void exec(const std::vector<MemoryCPtr>& srcMemory, const std::vector<MemoryCPtr>& dstMemory) override;
        void execSliceScatter(const std::vector<MemoryCPtr>& srcMemory, const std::vector<MemoryCPtr>& dstMemory);
        void execStridedSlice(const std::vector<MemoryCPtr>& srcMemory, const std::vector<MemoryCPtr>& dstMemory);

    private:
        struct StridedSliceParams {
            StridedSliceAttributes attrs;
            VectorDims srcBlockedDims;
            VectorDims srcOrder;
            VectorDims dstBlockedDims;
            VectorDims srcStrides;
            VectorDims dstStrides;
            size_t nDimsForWork = 0lu;
            bool isOptimized = false;
        };

        void paramsInitialization(const StridedSliceAttributes& attrs,
                                  const std::vector<MemoryCPtr>& srcMemory,
                                  const std::vector<MemoryCPtr>& dstMemory);
        void dimsNormalization();
        void dimsGluing();
        void indicesCalculation();
        void indicesCalculationForOptimized();
        void orderParametersByLayouts(const BlockedMemoryDescCPtr& blockedMemoryDesc);

        StridedSliceParams params;
        VectorDims srcIndices;
        VectorDims dstIndices;
        size_t nThreads = 0lu;
        size_t workAmount = 0lu;
        size_t lastDstDim = 0lu;
        size_t srcShift = 0lu;
        size_t m_threads_num = 0lu;
    };
    using executorPtr = std::shared_ptr<StridedSliceExecutor>;
    executorPtr execPtr = nullptr;

    bool isStrideSpecified = false;
    bool isAxesSpecified = false;

    bool isConstantInput[6] = {false};
    bool shapeHasDataDependency = false;
    bool hasConstAttrInputs = true;

    std::vector<MemoryCPtr> srcMemory;
    std::vector<MemoryCPtr> dstMemory;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
