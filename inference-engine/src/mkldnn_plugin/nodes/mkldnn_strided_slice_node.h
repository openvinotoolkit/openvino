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

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    inline void stridedSlice();

    void addHiddenDims(const size_t nSrcDims);
    void orderParametersByLayouts();
    void dimsNormalization(InferenceEngine::SizeVector& newSrcDims, InferenceEngine::SizeVector& newDstDims);
    void dimsGluing(const size_t realNDims, const InferenceEngine::SizeVector& newSrcDims, const InferenceEngine::SizeVector& newDstDims);
    void indicesCalculation();
    void indicesCalculationForOptimized();

    const size_t DATA_ID = 0;
    const size_t BEGIN_ID = 1;
    const size_t END_ID = 2;
    const size_t STRIDE_ID = 3;

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

    struct {
        MKLDNNMemoryPtr srcMemPtr = nullptr;
        MKLDNNMemoryPtr dstMemPtr = nullptr;
        InferenceEngine::SizeVector srcDims;
        InferenceEngine::SizeVector dstDims;
        InferenceEngine::SizeVector srcStrides;
        InferenceEngine::SizeVector dstStrides;
        InferenceEngine::SizeVector srcIndices;
        InferenceEngine::SizeVector dstIndices;
        int ellipsisPos1 = -1;
        int ellipsisPos2 = 0;
        size_t nThreads = 0;
        size_t nDimsForWork = 0;
        size_t workAmount = 0;
        size_t lastDstDim = 0;
        size_t dataSize = 0;
        size_t srcShift = 0;
        bool isOptimized = false;
        bool equalDims = false;
        bool parametersAreConstant = true;
    } params;
};

}  // namespace MKLDNNPlugin
