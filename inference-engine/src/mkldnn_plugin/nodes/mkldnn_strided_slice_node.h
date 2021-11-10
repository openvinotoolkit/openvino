// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNStridedSliceNode : public MKLDNNNode {
public:
    MKLDNNStridedSliceNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    void prepareParams() override;

protected:
    void executeDynamicImpl(mkldnn::stream strm) override;

private:
    void addHiddenDims(const size_t nSrcDims, int ellipsisPos1);
    void orderParametersByLayouts(const MKLDNNMemoryPtr& srcMemPtr);

    struct StridedSliceAttributes {
        std::vector<int> begin;
        std::vector<int> end;
        std::vector<int> stride;

        std::vector<int> beginMask;
        std::vector<int> endMask;
        std::vector<int> ellipsisMask;
        std::vector<int> newAxisMask;
        std::vector<int> shrinkAxisMask;

        InferenceEngine::SizeVector beginDims;
        InferenceEngine::SizeVector endDims;
        InferenceEngine::SizeVector strideDims;

        bool equalDims = false;
        size_t dataSize = 1lu;
    } attrs;

    struct StridedSliceExecutor {
        StridedSliceExecutor(const StridedSliceAttributes& attrs, const InferenceEngine::SizeVector& srcDims, const InferenceEngine::SizeVector& dstDims);
        void exec(const uint8_t* srcData, uint8_t* dstData);
        ~StridedSliceExecutor() = default;

    private:
        void dimsNormalization(InferenceEngine::SizeVector& newSrcDims, InferenceEngine::SizeVector& newDstDims);
        void dimsGluing(const size_t realNDims, const InferenceEngine::SizeVector& newSrcDims, const InferenceEngine::SizeVector& newDstDims);
        void indicesCalculation();
        void indicesCalculationForOptimized();

        struct {
            StridedSliceAttributes attrs;

            InferenceEngine::SizeVector srcDims;
            InferenceEngine::SizeVector dstDims;
            InferenceEngine::SizeVector srcStrides;
            InferenceEngine::SizeVector dstStrides;
            InferenceEngine::SizeVector srcIndices;
            InferenceEngine::SizeVector dstIndices;

            size_t nThreads = 0lu;
            size_t nDimsForWork = 0lu;
            size_t workAmount = 0lu;
            size_t lastDstDim = 0lu;
            size_t srcShift = 0lu;

            bool isOptimized = false;
        } params;
    };
    using executorPtr = std::shared_ptr<StridedSliceExecutor>;
    executorPtr execPtr = nullptr;

    bool isStrideSpecified = false;

    static constexpr size_t DATA_ID = 0;
    static constexpr size_t BEGIN_ID = 1;
    static constexpr size_t END_ID = 2;
    static constexpr size_t STRIDE_ID = 3;
};

}  // namespace MKLDNNPlugin
