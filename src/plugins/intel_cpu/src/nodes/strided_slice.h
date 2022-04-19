// Copyright (C) 2018-2022 Intel Corporation
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
    StridedSlice(const std::shared_ptr<ov::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    bool isExecutable() const override;

protected:
    void prepareParams() override;
    void executeDynamicImpl(dnnl::stream strm) override;

private:
    void addHiddenDims(const size_t nSrcDims, int ellipsisPos1);
    void orderParametersByLayouts(const MemoryPtr& srcMemPtr);

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
    } attrs;

    struct StridedSliceExecutor {
        StridedSliceExecutor(const StridedSliceAttributes& attrs, const VectorDims& srcBlockedDims, const VectorDims& dstBlockedDims);
        void exec(const uint8_t* srcData, uint8_t* dstData);
        ~StridedSliceExecutor() = default;

    private:
        struct StridedSliceParams {
            StridedSliceAttributes attrs;
            VectorDims srcBlockedDims;
            VectorDims dstBlockedDims;
            VectorDims srcStrides;
            VectorDims dstStrides;
            size_t nDimsForWork = 0lu;
            bool isOptimized = false;
        };

        void dimsNormalization(StridedSliceParams& params);
        void dimsGluing(StridedSliceParams& params, const size_t realNDims);
        void indicesCalculation(const StridedSliceParams& params);
        void indicesCalculationForOptimized(const StridedSliceParams& params);

        VectorDims srcIndices;
        VectorDims dstIndices;
        size_t nThreads = 0lu;
        size_t workAmount = 0lu;
        size_t lastDstDim = 0lu;
        size_t srcShift = 0lu;
    };
    using executorPtr = std::shared_ptr<StridedSliceExecutor>;
    executorPtr execPtr = nullptr;

    bool isStridedSliceOp = true;
    bool isStrideSpecified = false;
    bool isAxesSpecified = false;

    static constexpr size_t DATA_ID = 0;
    static constexpr size_t BEGIN_ID = 1;
    static constexpr size_t END_ID = 2;
    static constexpr size_t STRIDE_ID = 3;
    static constexpr size_t AXES_ID = 4;

    bool isConstantInput[AXES_ID + 1] = {false};
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
