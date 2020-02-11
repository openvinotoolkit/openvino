// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNPoolingNode : public MKLDNNNode {
public:
    MKLDNNPoolingNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, int socket);
    ~MKLDNNPoolingNode() override = default;

    void createDescriptor(const std::vector<InferenceEngine::TensorDesc>& inputDesc,
                          const std::vector<InferenceEngine::TensorDesc>& outputDesc) override;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void initDescriptor(const InferenceEngine::LayerConfig &config) override;
    void createPrimitive() override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

private:
    void setPostOps(mkldnn::primitive_attr &attr, bool initWeights = false);

    InferenceEngine::PoolingLayer::PoolType type = InferenceEngine::PoolingLayer::MAX;
    bool exclude_pad = false;
    std::vector<ptrdiff_t> stride;
    std::vector<ptrdiff_t> paddingL;
    std::vector<ptrdiff_t> paddingR;
    std::vector<ptrdiff_t> kernel;

    InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP32;
    InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::FP32;

    std::vector<MKLDNNMemoryPtr> PostOpsIntBlobMemory;
};

}  // namespace MKLDNNPlugin

