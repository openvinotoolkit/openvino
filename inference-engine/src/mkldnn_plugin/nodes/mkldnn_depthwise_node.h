// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNDepthwiseNode : public MKLDNNNode {
public:
    MKLDNNDepthwiseNode(InferenceEngine::CNNLayerPtr layer, const mkldnn::engine& eng);
    ~MKLDNNDepthwiseNode() override = default;

    void createDescriptor(const std::vector<InferenceEngine::TensorDesc>& inputDesc,
                          const std::vector<InferenceEngine::TensorDesc>& outputDesc) override;
    void initOptimalPrimitiveDescriptor() override;
    void getSupportedDescriptors() override;
    void createPrimitive() override;
    bool created() const override;

    mkldnn::algorithm getAlgorithm() {
        if (!initialized)
            initValues();
        return algorithm;
    }

    bool isWithBiases() {
        if (!initialized)
            initValues();
        return withBiases;
    }

    bool isBroadcast() {
        if (!initialized)
            initValues();
        return broadcast;
    }

private:
    void initValues();
    bool initialized = false;

    static Register<MKLDNNDepthwiseNode> reg;

    mkldnn::algorithm algorithm;
    size_t realWeightSize = 0;
    size_t realBiasSize = 0;
    bool withBiases;
    bool broadcast;
};

}  // namespace MKLDNNPlugin
