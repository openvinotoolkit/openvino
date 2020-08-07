// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include "caseless.hpp"
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNActivationNode : public MKLDNNNode {
public:
    MKLDNNActivationNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNActivationNode() override = default;

    void getSupportedDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    void createDescriptor(const std::vector<InferenceEngine::TensorDesc>& inputDesc,
                          const std::vector<InferenceEngine::TensorDesc>& outputDesc) override;
    void createPrimitive() override;
    bool created() const override;

    mkldnn::algorithm getAlgorithm() const { return algorithm; }
    float getAlpha() const { return alpha; }
    float getBeta() const { return beta; }

    MKLDNNMemoryDesc getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
    MKLDNNMemoryDesc getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;

private:
    float alpha = 0.0f;
    float beta = 0.0f;
    static InferenceEngine::details::caseless_map<std::string,
            std::function<void(InferenceEngine::GenericLayer*, mkldnn::algorithm&, float&, float&)>> initializers;
    mkldnn::algorithm algorithm = mkldnn::algorithm::eltwise_relu;
};

}  // namespace MKLDNNPlugin

