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
    MKLDNNStridedSliceNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNStridedSliceNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

private:
    void stridedSliceV();
    void stridedSlice();

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

    struct {
        InferenceEngine::SizeVector srcDims;
        InferenceEngine::SizeVector dstDims;
        InferenceEngine::SizeVector srcStrides;
        InferenceEngine::SizeVector dstStrides;
        InferenceEngine::SizeVector srcIndices;
        InferenceEngine::SizeVector dstIndices;
        size_t nThreads = 0;
        size_t nDimsForWork = 0;
        size_t workAmount = 0;
        size_t lastDstDim = 0;
        size_t dataSize = 0;
        bool equalDims = false;
    } params;
};

}  // namespace MKLDNNPlugin
