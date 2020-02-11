// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include "details/caseless.hpp"
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNActivationNode : public MKLDNNNode {
public:
    MKLDNNActivationNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, int socket);
    ~MKLDNNActivationNode() override = default;

    void getSupportedDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    void createDescriptor(const std::vector<InferenceEngine::TensorDesc>& inputDesc,
                          const std::vector<InferenceEngine::TensorDesc>& outputDesc) override;
    void createPrimitive() override;
    bool created() const override;

    mkldnn::algorithm getAlgorithm() {
        if (!initialized)
            initValues();
        return algorithm;
    }

    float getAlpha() {
        if (!initialized)
            initValues();
        return alpha;
    }

    float getBeta() {
        if (!initialized)
            initValues();
        return beta;
    }

    MKLDNNMemoryDesc getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
    MKLDNNMemoryDesc getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;

private:
    void initValues();
    bool initialized = false;
    float alpha = 0.0f;
    float beta = 0.0f;
    static InferenceEngine::details::caseless_map<std::string,
            std::function<void(InferenceEngine::GenericLayer*, mkldnn::algorithm&, float&, float&)>> initializers;
    mkldnn::algorithm algorithm = mkldnn::algorithm::eltwise_relu;
};

}  // namespace MKLDNNPlugin

